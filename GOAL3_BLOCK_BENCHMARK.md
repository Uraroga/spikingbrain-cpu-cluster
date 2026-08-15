# Goal 3 — Benchmark dei blocchi sintetici completi

## Decisione

**GO — si può passare al Goal 4 e caricare per la prima volta una piccola partizione reale in modo controllato.**

La partizione raccomandata cambia da 16/12 a **14/14**:

- atlas5: embedding e layer `0..13`;
- argo3: layer `14..27`, norm finale e `lm_head`.

I blocchi completi, dominati dalle grandi matrici FP32 e dalla fake-quantizzazione, hanno prestazioni di decode sostanzialmente equivalenti sui due nodi. Il taglio 14/14 bilancia meglio tempi e RAM; assegnare 16 o 18 layer ad atlas5 crea invece un evidente collo di bottiglia sul primo stadio.

Questo GO autorizza soltanto un caricamento piccolo e controllato nel Goal 4. Non implica che l'inferenza completa sarà interattiva: la sola somma stimata dei 28 blocchi è circa 10,4 secondi per token, prima di `lm_head`, rete e overhead di orchestrazione.

## Immagine Docker

L'immagine finale è stata costruita una sola volta su atlas5 e trasferita con `docker save | ssh ... docker load`.

```text
spikingbrain-cpu:goal3
Image ID/RepoDigest:
sha256:84f0ef2062aebaf63a59b1022d9fca2769abffe9660d74ce7b5a26a9f4b3098c
```

ID e digest sono identici su atlas5 e argo3. Tutti i benchmark finali riportati sotto provengono da questa stessa build, con seed 2026, PyTorch 2.13.0+cpu, Python 3.12.3, 4 thread, un warmup e tre iterazioni in `torch.inference_mode()`.

## Struttura dei blocchi

`src/spikingbrain_cpu/block.py` implementa:

### `GLABlock` — layer pari

1. RMSNorm dell'input;
2. QuantLinear Q 3584→3584;
3. QuantLinear K, V e gate 3584→512;
4. ReLU su Q/K e `logsigmoid(gate)/16`;
5. espansione da 4 KV heads a 28 heads;
6. GLA ricorrente con stato cache `[batch,28,128,128]`;
7. RMSNorm per testa e QuantLinear O 3584→3584;
8. fused residual/prenorm;
9. MLP con QuantLinear gate/up 3584→18944, SwiGLU e down projection 18944→3584;
10. residual finale.

### `SlidingWindowAttentionBlock` — layer dispari

1. RMSNorm dell'input;
2. QuantLinear Q 3584→3584 e K/V 3584→512;
3. rotary embedding CPU, theta 1.000.000;
4. cache K/V a 4 teste, limitata a window+1;
5. causal grouped-query attention con left window 4096;
6. QuantLinear O, fused residual/prenorm e lo stesso MLP;
7. residual finale.

Le dimensioni coincidono con `config.json`: hidden 3584, intermediate 18944, 28 teste, 4 KV heads, head dimension 128, 28 layer e window 4096. `use_short_conv=false`, quindi ShortConvolution non fa parte del percorso target.

Un solo `QuantBuffer` FP32 viene condiviso sequenzialmente da tutte le proiezioni del blocco. La down projection MLP usa direttamente `down_proj.weight` dentro SwiGLU, replicando il percorso effettivo di `modeling_gla_swa.py`.

## Test funzionali

Risultato nella stessa immagine su entrambi i nodi:

```text
atlas5: 28 passed, 1 warning in 1.73s
argo3:  28 passed, 1 warning in 1.91s
```

I 13 nuovi test coprono blocchi e planner, oltre ai 15 operator test del Goal 2:

- shape, dtype FP32 e output finiti;
- cache GLA e KV con posizione progressiva;
- limite esatto della cache sliding-window;
- determinismo senza cache;
- equivalenza numerica fra prefill unico e decode token-per-token;
- residual invariato con proiezioni azzerate;
- unico buffer QuantLinear dimensionato alla proiezione maggiore;
- inferenza delle shape dall'indice senza shard;
- split dei tensori e controllo del totale byte;
- errori espliciti per tensor name o `total_size` incoerenti.

Il warning è quello già noto per NumPy opzionale non installato; nessun test o operatore del progetto usa NumPy.

## RAM dei blocchi real-size

Ogni tipo è stato creato in un container/processo separato; non sono mai esistiti due blocchi completi contemporaneamente.

### atlas5

| Misura | GLA | Attention |
|---|---:|---:|
| RSS prima della costruzione | ~211 MiB | ~212 MiB |
| Parametri e scale calcolati | **903,05 MiB** | **895,99 MiB** |
| RSS dopo costruzione | ~1.118 MiB | ~1.114 MiB |
| Buffer QuantLinear | **259,00 MiB** | **259,00 MiB** |
| Cache rappresentativa | **1,75 MiB** | **16,00 MiB** |
| Picco processo | **1.402,30 MiB** | **1.526,03 MiB** |

### argo3

| Misura | GLA | Attention |
|---|---:|---:|
| RSS prima della costruzione | ~211 MiB | ~211 MiB |
| Parametri e scale calcolati | **903,05 MiB** | **895,99 MiB** |
| Buffer QuantLinear | **259,00 MiB** | **259,00 MiB** |
| Cache rappresentativa | **1,75 MiB** | **16,00 MiB** |
| Picco processo | **1.406,24 MiB** | **1.526,59 MiB** |

Le dimensioni dei blocchi coincidono con le stime esatte dagli header del Goal 1: 0,881882 GiB per GLA e 0,874990 GiB per attention. Ciò conferma che il wrapper contiene l'insieme corretto di parametri e scale.

Il picco attention è circa 127 MiB sopra l'RSS finale soprattutto per ripetizione K/V da 4 a 28 heads, score/softmax e concatenazione cache. La cache da 4096 contiene soltanto K/V a 4 teste. Il buffer da 259 MiB è uno per processo, non uno per layer.

## Benchmark atlas5

Tempi mediani in millisecondi:

| Componente | GLA | Attention |
|---|---:|---:|
| QuantLinear gate 3584→18944, decode | **136,313** | **136,911** |
| core GLA/attention, decode | **1,847** | **48,718** |
| MLP completo, decode | **295,026** | **278,931** |
| blocco completo, decode cached | **361,894** | **399,406** |
| blocco completo, prefill L4 | **400,200** | **429,385** |

Il decode attention usa una cache K/V da 4096 token. Il decode GLA usa uno stato `[1,28,128,128]` con posizione logica 4096.

## Benchmark argo3

| Componente | GLA | Attention |
|---|---:|---:|
| QuantLinear gate 3584→18944, decode | **130,638** | **130,548** |
| core GLA/attention, decode | **1,953** | **37,027** |
| MLP completo, decode | **276,652** | **276,064** |
| blocco completo, decode cached | **341,327** | **378,126** |
| blocco completo, prefill L4 | **454,066** | **444,907** |

## Rapporto argo3/atlas5

| Componente | GLA | Attention |
|---|---:|---:|
| QuantLinear gate | 0,96× | 0,95× |
| core | 1,06× | 0,76× |
| MLP | 0,94× | 0,99× |
| blocco decode | **0,94×** | **0,95×** |
| blocco prefill L4 | **1,13×** | **1,04×** |

I rapporti sotto 1 non vanno interpretati come superiorità generale dell'i3: i benchmark hanno tre campioni, nessuna CPU affinity/governor isolation e sono memory-bound. La conclusione robusta è che, sulle matrici real-size, argo3 non è 2–4 volte più lento come suggerivano le matrici piccole del Goal 2.5; il decode completo è entro circa 6%, mentre il breve prefill è 4–13% più lento.

## Costo di QuantLinear nel blocco

Una sola gate projection da 259 MiB costa circa 131–137 ms per token. Il MLP, che esegue due proiezioni fake-quantizzate di quella dimensione più la down projection FP32, costa 277–295 ms. Il core GLA è meno dell'1% del blocco; anche l'attention core con cache 4096 è soltanto circa 10–12%.

Il vero collo di bottiglia resta quindi la fake-quantizzazione FP32 per-forward:

- divisione, rounding e riscalatura devono leggere/scrivere l'intera matrice;
- il buffer elimina grandi allocazioni temporanee, ma non il traffico memoria;
- non si ottiene né compressione dei pesi né GEMM int8;
- una futura ottimizzazione di QuantLinear avrebbe impatto molto maggiore di un'ottimizzazione del core GLA.

## Stima delle partizioni

Per ogni coppia consecutiva pari/dispari sono stati sommati i tempi reali dei due blocchi. Cache per coppia: 1,75 + 16 = 17,75 MiB. I margini RAM sottraggono pesi, un buffer da 259 MiB per nodo e le cache di tutti i layer; non sottraggono ancora runtime base, hidden temporanei o `lm_head` workspace, quindi sono stime ottimistiche.

### Decode cached, batch 1

| Split | Pesi atlas5 | Pesi argo3 | Margine atlas5 | Margine argo3 | Blocchi atlas5 | Blocchi argo3 | Collo probabile |
|---|---:|---:|---:|---:|---:|---:|---|
| **14/14** | 14,3284 GiB | 14,3284 GiB | ~14,30 GiB | ~15,30 GiB | **5.329 ms** | **5.036 ms** | atlas5, solo +5,8% |
| 16/12 | 16,0853 GiB | 12,5715 GiB | ~12,52 GiB | ~17,07 GiB | 6.090 ms | 4.317 ms | atlas5, +41% |
| 18/10 | 17,8421 GiB | 10,8147 GiB | ~10,75 GiB | ~18,85 GiB | 6.852 ms | 3.597 ms | atlas5, +90% |
| 20/8 | 19,5990 GiB | 9,0578 GiB | ~8,97 GiB | ~20,62 GiB | 7.613 ms | 2.878 ms | atlas5, +165% |

La latenza batch-1 attraversa entrambi gli stadi, quindi la somma dei soli blocchi è circa 10,37–10,49 secondi in tutti i tagli. La colonna “collo” rappresenta invece il tempo di servizio dello stadio e il potenziale throughput con pipeline/microbatch.

### Prefill breve L4

| Split | Blocchi atlas5 | Blocchi argo3 | Collo probabile |
|---|---:|---:|---|
| **14/14** | **5.807 ms** | **6.293 ms** | argo3, +8% |
| 16/12 | 6.637 ms | 5.394 ms | atlas5, +23% |
| 18/10 | 7.466 ms | 4.495 ms | atlas5, +66% |
| 20/8 | 8.296 ms | 3.596 ms | atlas5, +131% |

Anche nel prefill il 14/14 è il più vicino all'equilibrio. `lm_head`, che risiede su argo3, non è incluso e sposterà parte del costo verso il secondo stadio; embedding su atlas5 è poco costoso. Questo può rendere il 14/14 ancora più equilibrato nel decode, ma dovrà essere misurato con una partizione reale controllata.

## Loader selettivo: sola struttura

`src/spikingbrain_cpu/selective_loader.py`:

- apre esclusivamente `config.json` e `model.safetensors.index.json`;
- mappa ogni tensor name al relativo shard;
- inferisce shape FP32 da configurazione e nome del modulo;
- filtra embedding/layer range/norm/head per nodo;
- verifica che la somma dei byte inferiti coincida con `metadata.total_size`;
- stampa nome, shard, shape, dtype e byte di ogni tensore;
- non importa Safetensors e non contiene alcuna materializzazione dati.

Il dry-run è stato eseguito su argo3 montando in read-only soltanto i due JSON, mai gli shard. Senza `--summary-only` produce 596 righe, incluse esattamente le 591 righe tensoriali.

Risultati:

| Split | Tensori atlas5 | GiB atlas5 | Tensori argo3 | GiB argo3 | Totale |
|---|---:|---:|---:|---:|---:|
| 14/14 | 295 | 14,328382 | 296 | 14,328395 | 28,656777 GiB |
| 16/12 | 337 | 16,085255 | 254 | 12,571523 | 28,656777 GiB |
| 18/10 | 379 | 17,842127 | 212 | 10,814650 | 28,656777 GiB |
| 20/8 | 421 | 19,599000 | 170 | 9,057777 | 28,656777 GiB |

Esempio per il taglio raccomandato:

```bash
docker run --rm \
  -v /percorso/config.json:/checkpoint/config.json:ro \
  -v /percorso/model.safetensors.index.json:/checkpoint/index.json:ro \
  spikingbrain-cpu:goal3 \
  python -m spikingbrain_cpu.selective_loader \
    --dry-run --split-layer 14 \
    --config /checkpoint/config.json --index /checkpoint/index.json
```

## Nuovi bloccanti e rischi

1. **QuantLinear domina il tempo.** Il full model FP32 sarà lento anche se la distribuzione funziona.
2. **Attention prefill resta quadratica.** La reference implementation materializza gli score completi e non è adatta a prompt lunghi.
3. **GLA prefill usa loop Python.** L4 è gestibile, lunghezze elevate richiederanno chunking/compilazione CPU.
4. **`lm_head` non benchmarkato.** Sono 2,03 GiB FP32 su argo3 e deve entrare nel test Goal 4 prima dell'inferenza completa.
5. **Margini RAM ottimistici.** Il loader reale deve costruire su `meta`, caricare tensor-by-tensor e misurare RSS senza full state dict o copie temporanee.
6. **Nessuna equivalenza con kernel GPU.** I test validano matematica e cache del fallback, non bit-equivalence con Triton/FlashAttention.

Nessuno di questi rischi impedisce di aprire nel Goal 4 una piccola partizione reale sotto limiti di byte/RSS espliciti. Impediscono invece di passare direttamente al modello completo o alla generazione.

## Integrità dei dati

- Nessuno shard Safetensors è stato aperto, letto o montato.
- Nessun peso reale è stato caricato o copiato.
- `$HOME/Modelli/SpikingBrain` non è stata creata su atlas5.
- Il checkpoint originale non è stato modificato.
- Non sono stati implementati Gloo, inferenza distribuita, generazione o stage con pesi reali.

## Raccomandazione finale

Adottare **14/14** come nuova baseline. Nel Goal 4 caricare inizialmente un numero minimo di tensori o un singolo blocco su argo3, verificando shape, checksum logico, RSS e assenza di copie; non copiare ancora l'intero sottoinsieme atlas5. Solo dopo il benchmark reale di un blocco e di `lm_head` si potrà confermare definitivamente la partizione per l'esecuzione completa.
