# Goal 4 — Primo caricamento controllato di pesi reali

## Decisione

**GO — il loader reale è sicuro e si può iniziare a costruire una partizione completa su un nodo, in modo incrementale e con limiti RSS/byte obbligatori.**

Il test ha materializzato separatamente tre tensori, il solo layer GLA 14 e, in un nuovo processo, il solo `lm_head`. Non è mai esistita una partizione completa, un full state dict o una copia dei pesi su atlas5.

Il layer reale riproduce quasi esattamente tempi e RAM del blocco sintetico. `lm_head` aggiunge circa 109 ms per token su argo3 e occupa 2,03 GiB, senza duplicazioni dense inattese.

## Ambiente

Immagine costruita una volta su atlas5 e trasferita con `docker save | ssh ... docker load`:

```text
spikingbrain-cpu:goal4
Image ID/RepoDigest:
sha256:929ae0ea3cff10e1c8dc89e992a00c227934762d1aa5f62c663e7615c76e220c
```

Digest identico sui due nodi. Dipendenze nuove: soltanto `safetensors==0.6.2`; restano assenti CUDA, Triton, `fla`, FlashAttention e NumPy.

Test sintetici Docker prima dell'accesso al checkpoint:

```text
31 passed, 1 warning in 1,75s
```

Tutte le esecuzioni reali sono avvenute su argo3 con:

- checkpoint montato nel container con bind mount `readonly`;
- `--network none`;
- PyTorch 2.13.0+cpu, FP32, 4 thread;
- processi separati per piccoli tensori, layer e `lm_head`;
- limiti espliciti `--max-bytes` e `--max-rss-mib`.

## Strategia del loader reale

`src/spikingbrain_cpu/selective_loader.py` ora contiene `RealTensorLoader`:

1. legge soltanto `config.json` e `model.safetensors.index.json` per pianificare;
2. verifica preventivamente tensor name, shape, dtype e byte inferiti;
3. raggruppa soltanto i nomi richiesti per shard;
4. apre unicamente quegli shard con `safetensors.safe_open(..., framework="pt", device="cpu")`;
5. recupera ogni tensore singolarmente con `get_tensor(name)`;
6. usa lo storage mmap restituito da Safetensors, senza clonarlo;
7. assegna direttamente il tensore al parametro/buffer corrispondente del modulo costruito su device `meta`;
8. non costruisce dizionari di tensori o state dict;
9. mantiene contatori cumulativi e interrompe prima dell'apertura se `--max-bytes` verrebbe superato;
10. controlla sia RSS corrente sia HWM contro `--max-rss-mib` prima e durante il caricamento.

Il test unitario verifica anche che il `data_ptr` del parametro assegnato sia uguale a quello del tensore sorgente: non viene introdotta una copia nell'assegnazione. Un test negativo con limite RSS 1 MiB termina prima di leggere tensori con:

```text
LoaderLimitError: RSS limit exceeded before loading:
rss=209,26 MiB, peak=209,26 MiB, limit=1,00 MiB
```

## Shard effettivamente aperti

| Test | Shard aperti |
|---|---|
| tre tensori selettivi | `model-00004-of-00007.safetensors` |
| layer 14 completo | `model-00004-of-00007.safetensors` |
| `lm_head.weight` | `model-00007-of-00007.safetensors` |

Nessuno degli shard 1, 2, 3, 5 o 6 è stato aperto. Lo shard 4 non è stato caricato interamente: dopo l'assegnazione mmap dei 23 tensori del layer, RSS era soltanto 216 MiB; le pagine sono diventate residenti solo quando toccate/lette.

## Piccolo insieme di tensori

I tre oggetti sono stati caricati e rilasciati uno alla volta nello stesso processo. Il “touch” legge un valore FP32 per pagina da 4 KiB per rendere osservabile la memoria mmap senza creare una copia densa.

| Tensor name | Shape | Byte | MiB | RSS prima | RSS mappato | RSS dopo touch | RSS dopo release |
|---|---:|---:|---:|---:|---:|---:|---:|
| `model.layers.14.attn_norm.weight` | `[3584]` | 14.336 | 0,014 | 210,72 | 213,88 | 215,35 | 215,29 |
| `model.layers.14.attn.k_proj.weight` | `[512,3584]` | 7.340.032 | 7,00 | 215,29 | 215,35 | 222,41 | 215,29 |
| `model.layers.14.mlp.gate_proj.weight` | `[18944,3584]` | 271.581.184 | 259,00 | 215,29 | 215,35 | 475,08 | 215,89 |

Tutti sono `torch.float32`; numero elementi e byte coincidono esattamente con il dry-run Goal 3. La grande QuantLinear aumenta RSS di 259,79 MiB quando le pagine vengono faultate e ritorna quasi esattamente alla baseline dopo unmap/gc. Non rimane una seconda copia del peso.

## Layer reale 14

È stato scelto il layer 14, di tipo GLA e appartenente alla futura partizione argo3.

### Materializzazione

| Campo | Valore |
|---|---:|
| Tensor count | 23 |
| Byte materializzati | 946.913.792 |
| Peso logico | 903,05 MiB / 0,881882 GiB |
| Shard | soltanto shard 4 |
| Tempo assegnazione mmap | 5,075 ms |
| Limite byte | 1.000.000.000 |
| Limite RSS | 2.500 MiB |

Il wrapper è stato creato su `meta`, quindi la costruzione non ha inizializzato pesi sintetici né allocato un secondo layer. Ogni parametro/buffer meta è stato sostituito direttamente dal tensore Safetensors corrispondente. Al termine non rimanevano tensori meta.

### RAM

| Punto | RSS | HWM |
|---|---:|---:|
| processo base | 210,61 MiB | 210,61 MiB |
| wrapper meta | 212,12 MiB | 212,12 MiB |
| dopo assegnazione mmap | 215,98 MiB | 215,98 MiB |
| dopo page touch di tutti i pesi | 1.119,25 MiB | 1.119,25 MiB |
| dopo buffer QuantLinear | 1.378,48 MiB | 1.378,48 MiB |
| dopo forward/benchmark | 1.397,69 MiB | **1.400,49 MiB** |
| dopo release layer/process objects | 234,10 MiB | 1.400,49 MiB |

Componenti persistenti osservati:

- pesi e scale reali: 903,05 MiB;
- un solo buffer fake-quant: 259,00 MiB;
- cache GLA: 1,75 MiB;
- temporanei/allocator durante forward: circa 22 MiB sopra pesi+buffer+cache.

Il picco Goal 3 per lo stesso blocco sintetico era 1.406,24 MiB su argo3; il reale è 1.400,49 MiB. Non c'è duplicazione inattesa.

### Validazione forward

Input sintetico FP32 `[1,1,3584]`, cache GLA `[1,28,128,128]` a posizione 4096:

```text
output shape:  [1, 1, 3584]
output dtype:  torch.float32
finite:        true
cache shape:   [1, 28, 128, 128]
new position:  4097
```

Non sono stati usati tokenizer, logits linguistici o generazione.

## Benchmark layer reale

Un warmup, tre iterazioni, `torch.inference_mode()`, 4 thread:

| Componente | Sintetico Goal 3 | Reale Goal 4 | Reale/sintetico |
|---|---:|---:|---:|
| QuantLinear gate 3584→18944 | 130,638 ms | **130,244 ms** | 1,00× |
| GLA core | 1,953 ms | **2,403 ms** | 1,23× |
| MLP | 276,652 ms | **275,148 ms** | 0,99× |
| blocco completo cached | 341,327 ms | **340,567 ms** | 1,00× |

La differenza del core è 0,45 ms e meno dello 0,2% del blocco; è compatibile con jitter e valori input diversi. Totale, QuantLinear e MLP coincidono entro l'1%. I pesi reali non introducono costi o instabilità numeriche rispetto alla previsione sintetica.

## `lm_head` reale

Il test è stato eseguito dopo la completa terminazione del container layer, in un nuovo processo.

| Campo | Valore |
|---|---:|
| Tensor | `lm_head.weight` |
| Shard | `model-00007-of-00007.safetensors` |
| Shape | `[152064,3584]` |
| Elementi | 544.997.376 |
| Byte | 2.179.989.504 |
| Dimensione | 2.079 MiB / 2,030273 GiB |
| Dtype | FP32 |
| Limite byte | 2.200.000.000 |
| Limite RSS | 4.000 MiB |

### RAM `lm_head`

| Punto | RSS/HWM |
|---|---:|
| processo base | 210,71 MiB |
| dopo assegnazione mmap | 213,96 MiB |
| dopo page touch | 2.294,79 MiB |
| dopo proiezione | 2.301,59 MiB |
| picco | **2.302,54 MiB** |
| dopo release | 224,05 MiB |

L'aumento dopo page touch corrisponde al peso da 2.079 MiB. La proiezione aggiunge soltanto circa 6,8 MiB tra logits e allocator; non appare una copia completa di `lm_head`.

### Tempo `lm_head`

Input `[1,3584]`, un warmup e tre iterazioni:

| Operazione | Mediana | Min–max |
|---|---:|---:|
| proiezione 3584→152064 | **109,436 ms** | 109,212–109,496 ms |
| argmax | **0,338 ms** | 0,336–0,350 ms |
| top-5 | **0,526 ms** | 0,511–1,327 ms |

Logits: `[1,152064]`, FP32, 608.256 byte, tutti finiti. Nessun loop autoregressivo è stato avviato.

## Tabella RAM riassuntiva

| Oggetto | Pesi | RSS base | RSS caricato/toccato | RSS picco | Temporanei principali |
|---|---:|---:|---:|---:|---|
| QuantLinear grande | 259,00 MiB | 215,29 | 475,08 | 475,08 MiB | mmap metadata trascurabile; nessuna copia residua |
| layer GLA 14 | 903,05 MiB | 210,61 | 1.119,25 | 1.400,49 MiB | buffer 259 MiB, cache 1,75 MiB, ~22 MiB forward |
| `lm_head` | 2.079 MiB | 210,71 | 2.294,79 | 2.302,54 MiB | logits 0,58 MiB e ~6 MiB allocator |

Le misure confermano le stime Goal 3. La strategia mmap rende il costo di “assegnazione” quasi nullo finché le pagine non sono lette, ma la RAM fisica richiesta durante l'uso resta naturalmente pari ai pesi residenti.

## Assenza di full state dict e copie

- Il loader non chiama `state_dict()`, `load_state_dict()`, `torch.load()` o `from_pretrained()`.
- Il generatore conserva un solo tensore corrente durante l'assegnazione.
- Il modulo conserva i 23 tensor storage mmap richiesti, non un dizionario aggiuntivo.
- L'assegnazione diretta preserva il `data_ptr`.
- Il buffer QuantLinear è l'unico duplicato denso intenzionale e riusabile; contiene al massimo una matrice da 259 MiB.
- Le riduzioni di page-touch campionano una word per pagina e non clonano il tensore.
- Il ritorno RSS dopo release dimostra che layer e `lm_head` vengono unmappati.

## Integrità del checkpoint

Prima dei test sono stati registrati nome, size e mtime di tutti i 27 file della directory; lo snapshot successivo è byte-per-byte identico. Sono rimasti invariati anche:

```text
config.json:
6f0843853d2686808dea19fe715d4958dcff0c7007da587821b1890ed4bd8466

model.safetensors.index.json:
9e73b824a32963d5d1f439bac4f0a45e3a76005dcbdf13689832d751d8238f88
```

Inoltre:

- il bind mount Docker era read-only;
- il container aveva rete disabilitata;
- nessun file è stato creato nella directory modello;
- nessuno shard è stato duplicato;
- nessun peso è stato trasferito o creato su atlas5;
- `$HOME/Modelli/SpikingBrain` continua a non esistere su atlas5.

## Raccomandazione

Procedere nel Goal 5 alla costruzione incrementale della partizione argo3 14/14, mantenendo:

1. modulo su `meta`;
2. assegnazione tensor-by-tensor mmap;
3. whitelist esatta di prefissi;
4. limite byte calcolato prima di aprire shard;
5. limite HWM/RSS con stop immediato;
6. un solo buffer QuantLinear per processo;
7. caricamento di `lm_head` solo dopo aver verificato il margine RSS dei 14 layer;
8. nessuna copia o stage atlas5 finché argo3 non è validato.

Il costo previsto del secondo stadio aumenta di circa 109 ms/token per `lm_head`, oltre ai circa 5,04 secondi dei 14 blocchi stimati nel Goal 3. È lento ma coerente con l'obiettivo sperimentale; RAM e loader non richiedono un cambio di strategia prima del prossimo passo.
