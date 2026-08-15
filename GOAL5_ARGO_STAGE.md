# Goal 5 — Costruzione incrementale dello stage reale argo3

## Decisione

**GO — argo3 può mantenere stabilmente la propria metà reale del modello e si può iniziare il Goal 6 per costruire la metà atlas5.**

Lo stage completo layer 14–27 + norm + `lm_head` occupa al picco **15.805,24 MiB RSS/HWM**, sotto il limite conservativo di 20.000 MiB. Durante la permanenza dello stage restavano **11.188,80 MiB realmente liberi** e 29.691,86 MiB disponibili includendo cache reclaimable; nessuna pagina del processo è entrata in swap.

Il percorso completo sintetico-hidden → 14 layer reali → norm → `lm_head` → argmax produce logits finiti in **5.240,03 ms** e token tecnico 88055. Il benchmark dei soli layer ha mediana **5.185,03 ms**, circa +3,0% rispetto alla previsione Goal 3 di 5,04 s.

## Ambiente Docker

Immagine costruita una volta su atlas5 e trasferita senza ricostruzione:

```text
spikingbrain-cpu:goal5
Image ID/RepoDigest:
sha256:ae50f84a01fbb5d675b0cee586036185573de51cfc8e275d05210fdb75dab4c8
```

Digest identico su atlas5 e argo3. Test prima del caricamento:

```text
35 passed, 1 warning in 1,78s
```

Il container argo3 è stato eseguito con checkpoint bind-mounted `readonly`, `--network none`, PyTorch CPU 2.13.0, FP32, 4 thread e questi limiti:

| Limite | Valore |
|---|---:|
| byte materializzabili | 16.000.000.000 |
| RSS/HWM processo | 20.000 MiB |
| RAM host disponibile minima | 6.144 MiB |
| byte esatti pianificati | 15.384.997.376 |

Nessun limite è stato avvicinato o superato.

## Struttura `ArgoStage`

`src/spikingbrain_cpu/model_partition.py` introduce `ArgoStage`:

- `ModuleDict` con indici globali stringa `14`…`27`;
- `GLABlock` per layer pari e `SlidingWindowAttentionBlock` per layer dispari;
- tutti i parametri inizialmente su device `meta`;
- norm finale meta separato;
- `lm_head` meta separato e opzionale;
- metodi indipendenti `load_layer`, `load_final_norm`, `load_lm_head`;
- un unico `QuantBuffer` esterno/condiviso fra tutti i blocchi;
- cache indicizzate con layer globali, mai con offset locale;
- forward limitato ai layer e metodi separati per norm/logits.

Durante il controllo del percorso originale è stata corretta una discrepanza del wrapper attention Goal 3: dopo RoPE, Q passa ora da dynamic spikes e l'intera K/V cache passa da fake quantizzazione simmetrica 8-bit prima dell'attention, mentre la cache conserva K/V non quantizzate. Questo replica l'ordine effettivo di `window_attention.py`; non è un'ottimizzazione per il benchmark.

## Caricamento incrementale

I pesi di ogni nuovo layer sono stati assegnati direttamente da mmap e poi letti a passo pagina, affinché RSS misurasse pagine residenti reali. La colonna “delta RSS” usa come baseline lo stage meta a 217,14 MiB.

| Layer caricati | Ultimo layer globale | Tensori | Byte logici | MiB logici | RSS | HWM | Delta RSS | Shard unici | Open events |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 14 | 23 | 946.913.792 | 903,05 | 1.122,64 | 1.122,64 | 905,50 | 4 | 1 |
| 2 | 15 | 42 | 1.886.427.648 | 1.799,04 | 2.018,65 | 2.018,65 | 1.801,51 | 4 | 2 |
| 4 | 17 | 84 | 3.772.855.296 | 3.598,08 | 3.817,86 | 3.817,86 | 3.600,72 | 4 | 4 |
| 8 | 21 | 168 | 7.545.710.592 | 7.196,15 | 7.416,24 | 7.416,24 | 7.199,10 | 4, 5 | 9 |
| 14 | 27 | 294 | 13.204.993.536 | 12.593,26 | 12.813,93 | 12.813,93 | 12.596,79 | 4, 5, 6 | 16 |

Ad ogni checkpoint i byte caricati coincidono esattamente con il planner. La differenza delta RSS – pesi logici è soltanto circa 2–4 MiB: non appare una copia densa dei layer. I circa 217 MiB restanti sono runtime Python/PyTorch e struttura meta di processo.

Gli open events sono maggiori degli shard unici perché il loader apre e chiude lo shard per ciascun layer; i layer 18 e 23 attraversano un confine shard e generano un evento aggiuntivo. Nessuno shard non necessario è stato aperto durante i soli layer.

### Tempo di caricamento

Sulla build finale, con pagine già presenti nella cache filesystem per la precedente validazione, caricamento mmap + page touch cumulativo:

| Layer | Tempo cumulativo warm-cache |
|---:|---:|
| 1 | 17,75 ms |
| 2 | 35,54 ms |
| 4 | 71,53 ms |
| 8 | 141,09 ms |
| 14 | 253,74 ms |

La prima esecuzione prima del warm filesystem cache richiese **47,64 s** per tutti i layer. Il loader mmap stesso è rapido; la differenza è il page fault/I/O. La misura warm-cache non va usata come previsione di cold start dopo reboot.

## Shard aperti

| Contenuto | Shard |
|---|---|
| layer 14–27 | `model-00004-of-00007`, `00005`, `00006` |
| norm finale | shard 6 |
| `lm_head` | shard 7 |

Totale stage: quattro shard unici e 18 aperture controllate. Gli shard 1–3 non sono stati aperti.

## QuantBuffer condiviso

Verifica programmatica:

```text
numero identità QuantBuffer fra 14 layer: 1
dimensione: 271.581.184 byte = 259,00 MiB
```

| Punto | RSS |
|---|---:|
| prima del buffer | 12.813,93 MiB |
| dopo allocazione e page commit | 13.073,05 MiB |
| incremento | 259,12 MiB |

Non esiste un buffer per layer. La dimensione corrisponde alla proiezione 18944×3584 FP32 massima.

## Cache per layer

Sono state create 14 cache indipendenti a posizione logica 4096:

- layer pari 14,16,…,26: sette `GLACache` `[1,28,128,128]`;
- layer dispari 15,17,…,27: sette `KVCache`, ciascuna con K e V `[1,4,4096,128]`;
- nessun oggetto o storage cache condiviso;
- chiavi del dizionario esattamente 14…27.

| Misura | Valore |
|---|---:|
| cache count | 14 |
| byte totali | 130.285.568 |
| dimensione | 124,25 MiB |
| RSS prima | 13.073,05 MiB |
| RSS dopo | 13.197,43 MiB |
| posizione prima | 4096 per tutti |
| posizione dopo forward | 4097 per tutti |

## Primo forward layer 14–27

Input `[1,1,3584]`, FP32. Ogni layer ha restituito `[1,1,3584]`, FP32, senza NaN/Inf.

| Layer | Tipo | Tempo |
|---:|---|---:|
| 14 | GLA | 358,779 ms |
| 15 | attention | 404,027 ms |
| 16 | GLA | 341,158 ms |
| 17 | attention | 398,811 ms |
| 18 | GLA | 341,153 ms |
| 19 | attention | 399,276 ms |
| 20 | GLA | 341,183 ms |
| 21 | attention | 398,941 ms |
| 22 | GLA | 341,431 ms |
| 23 | attention | 397,967 ms |
| 24 | GLA | 340,540 ms |
| 25 | attention | 399,641 ms |
| 26 | GLA | 340,289 ms |
| 27 | attention | 398,455 ms |

Totale prima validazione: **5.203,72 ms**. Medie:

- GLA: 343,505 ms;
- attention: 399,588 ms.

RSS dopo il forward era 13.335,21 MiB; HWM 13.470,58 MiB. Tutte le cache sono avanzate di un token.

## Benchmark reale dei 14 layer

Un warmup e tre iterazioni, stesse cache iniziali read-only e stesso input:

| Statistica | Tempo |
|---|---:|
| mediana | **5.185,027 ms** |
| minimo | 5.131,309 ms |
| massimo | 5.198,823 ms |
| previsione Goal 3 | ~5.036 ms |
| scostamento mediana | **+2,96%** |

HWM dopo il benchmark: 13.725,43 MiB. La previsione Goal 3 era accurata; il piccolo aumento include il percorso Q/K/V fake-quant attention ora allineato al codice originale.

## Norm finale

`model.norm.weight` è stato caricato soltanto dopo la validazione e il benchmark dei 14 layer.

| Campo | Valore |
|---|---:|
| load + touch | 1,064 ms |
| tempo norm mediano | 0,082 ms |
| shape | `[1,1,3584]` |
| dtype | FP32 |
| output finito | sì |
| incremento RSS osservabile | ~0,07 MiB |

## `lm_head`

Prima del caricamento:

| Misura | Valore |
|---|---:|
| RSS processo | 13.590,16 MiB |
| HWM | 13.725,43 MiB |
| MemFree host | 10.963,28 MiB |
| MemAvailable host | 29.466,35 MiB |

Il margine superava sia i 2.079 MiB del peso sia la riserva conservativa di 1 GiB, quindi il caricamento è proseguito.

| Campo | Valore |
|---|---:|
| peso | 2.179.989.504 byte / 2.079 MiB / 2,030273 GiB |
| RSS dopo load + touch | 15.669,16 MiB |
| incremento | 2.079,00 MiB |
| proiezione mediana | 109,450 ms |
| logits | `[1,1,152064]`, FP32, finiti |
| argmax | 0,340 ms; token 88055 |
| top-5 | 0,515 ms |
| indici top-5 | 88055, 116989, 18522, 10501, 103678 |

L'aumento RSS coincide esattamente con la dimensione del peso: nessun clone di `lm_head`.

## Stage completo

Eseguito una volta:

```text
hidden sintetico
→ layer reali 14..27
→ norm finale reale
→ lm_head reale
→ argmax
```

| Misura | Valore |
|---|---:|
| tempo totale | **5.240,031 ms** |
| logits shape | `[1,1,152064]` |
| logits finiti | sì |
| token tecnico | 88055 |
| RSS finale | 15.669,60 MiB |
| HWM massimo | **15.805,24 MiB** |
| limite HWM | 20.000 MiB |
| margine dal limite | 4.194,76 MiB |

## RAM host residua e swap

Con lo stage ancora vivo al punto finale:

| Misura host/processo | Prima | Finale |
|---|---:|---:|
| MemFree host | 12.022,22 MiB | **11.188,80 MiB** |
| MemAvailable host | 30.525,14 MiB | **29.691,86 MiB** |
| Cached host | 18.585,84 MiB | 18.585,84 MiB |
| swap processo | 0 MiB | **0 MiB** |
| SwapFree host | 8.192 MiB | **8.192 MiB** |

`MemAvailable` include pagine file-backed reclaimable, quindi il valore prudente è MemFree: rimangono circa **10,93 GiB realmente liberi**, oltre 4,19 GiB sotto il limite RSS scelto. Nessuno swap è stato usato intenzionalmente o automaticamente dal processo.

## Loader e mapping finali

| Campo | Valore |
|---|---:|
| tensori caricati | **296** |
| byte logici stage | **15.384.997.376** |
| byte contatore loader | **15.384.997.376** |
| shard unici | 4, 5, 6, 7 |
| full state dict | assente |

I due contatori indipendenti coincidono con il piano dry-run del Goal 3 per il taglio 14/14.

## Integrità e vincoli

Prima e dopo l'esecuzione sono stati confrontati nome, size e mtime di tutti i 27 file del checkpoint: snapshot identico. Sono invariati anche gli hash:

```text
config.json:
6f0843853d2686808dea19fe715d4958dcff0c7007da587821b1890ed4bd8466

model.safetensors.index.json:
9e73b824a32963d5d1f439bac4f0a45e3a76005dcbdf13689832d751d8238f88
```

Inoltre:

- mount checkpoint read-only e rete container disabilitata;
- nessun `state_dict`, `load_state_dict`, `torch.load` o `from_pretrained`;
- nessun clone permanente dei pesi mmap;
- un solo duplicato intenzionale: buffer fake-quant riusabile da 259 MiB;
- nessun CUDA, Triton, `fla` o FlashAttention;
- nessun Gloo, networking del modello, tokenizer o generazione;
- nessun peso copiato su atlas5;
- `$HOME/Modelli/SpikingBrain` continua a non esistere su atlas5.

## Raccomandazione finale

Procedere al Goal 6 per costruire la metà atlas5 0–13. La partizione 14/14 è confermata sia da prestazioni sia da RAM. Per atlas5 mantenere la stessa disciplina:

1. copiare soltanto i dati autorizzati nel Goal 6, mai l'intero checkpoint implicitamente;
2. verificare checksum e mapping prima del load;
3. stage meta e mmap tensor-by-tensor;
4. checkpoint RSS 1/2/4/8/14;
5. un solo QuantBuffer;
6. limite RSS conservativo almeno 4 GiB sotto la capacità osservata;
7. nessun avvio distribuito finché AtlasStage non passa test e benchmark locali.

Argo3 può sostenere stabilmente lo stage reale, ma il margine di circa 11 GiB liberi non autorizza l'aggiunta di copie dense, full state dict o secondo buffer.
