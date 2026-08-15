# Goal 8.6 — Runtime PyTorch CPU OpenBLAS Ivy Bridge

## Decisione

**GO-RUNTIME-STABLE**: la nuova build ha completato tutti i test con **0 SIGILL**. Sono passati 50 processi indipendenti `exp/logsigmoid`, 30 processi ArgoStage reali e 5 sessioni distribuite tecniche indipendenti. Non sono state eseguite generazioni testuali.

Il runtime è sensibilmente più lento della build MKL, ma è sufficientemente stabile per ripetere Goal 8. L'immagine precedente `spikingbrain-cpu:goal8` è rimasta immutata e disponibile.

## 1. Sorgente e build PyTorch

- tag: `v2.13.0`;
- commit verificato: `cf30153c4c131c8164ee7798e5022d810682e2cb`;
- wheel: `torch-2.13.0+openblas.ivybridge`;
- Python: 3.12.3;
- build: Release, wheel multi-stage;
- immagine base pin tramite digest: `python:3.12.3-slim-bookworm@sha256:fd3817f...`;
- parallelismo di compilazione effettivo: `MAX_JOBS=4`;
- toolchain presente solo nello stage builder.

Artefatti riproducibili:

- `Dockerfile.safe-runtime`;
- `requirements.safe-runtime.lock`;
- `scripts/diagnose_sigill.py`;
- `scripts/diagnose_atlas_stage.py`.

La prima compilazione si era fermata perché il target PyTorch `gen_torch_version` richiedeva il modulo Python `packaging`. È stato aggiunto il pin builder `packaging==25.0`; non è stata modificata alcuna sorgente PyTorch.

## 2. Opzioni e target ISA

```text
USE_CUDA=0
USE_MKL=0
USE_MKLDNN=0
BLAS=OpenBLAS
USE_DISTRIBUTED=1
USE_GLOO=1
USE_MPI=0
USE_NCCL=0
BUILD_TEST=0
CFLAGS=-O2 -march=ivybridge -mtune=ivybridge
CXXFLAGS=-O2 -march=ivybridge -mtune=ivybridge
```

Non è stato usato `-march=native`. `torch.__config__.show()` conferma `BUILD_TYPE=Release`, `BLAS_INFO=open`, `LAPACK_INFO=open`, `USE_GLOO=1`, `USE_MKLDNN=0` e i flag Ivy Bridge. La configurazione CUDA riportata come campo informativo non implica codice CUDA: `USE_CUDA=0` e `torch.cuda.is_available()` è falso.

## 3. Backend BLAS e assenza MKL

Il runtime finale installa `libopenblas0-pthread`; `ldd libtorch_cpu.so` mostra il collegamento diretto a `libopenblas.so.0` e `libgomp`, senza librerie MKL.

| controllo | risultato |
|---|---|
| `torch.backends.mkldnn.is_available()` | `False` |
| `torch.backends.mkl.is_available()` | `False` |
| `BLAS_INFO` / `LAPACK_INFO` | `open` / `open` |
| simbolo/stringa esatta `mkl_vml_kernel_sExp_Z0HAynn` | assente |
| CUDA | assente/non disponibile |

`libtorch_cpu.so`:

- path: `/usr/local/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so`;
- SHA-256: `16bb4aa6dec59c9f6aae7c1dbc71760cb5ec4d9f2ea4def2526e9a83218f15d0`;
- ELF build-id: `9e4f77abdeb2091c92cb06f9082f6625e4e0d066`;
- non stripped.

Il gate fondamentale è quindi superato: il kernel MKL VML che causava l'AVX-512 illegale non è incorporato nel nuovo `libtorch_cpu.so`.

## 4. Immagine identica sui due nodi

Tag: `spikingbrain-cpu:goal8.6-openblas`

```text
Image ID: sha256:696d98683cee6eef018a73e5c70b402f9103fc24b230356975a7ef1685aed836
Size:     368007299 byte
```

L'immagine è stata costruita una sola volta su atlas5 e trasferita con `docker save | ssh <USER>@192.168.1.64 docker load`. Argo3 riporta lo stesso ID e la stessa dimensione. La baseline `spikingbrain-cpu:goal8`, ID `sha256:81ac4c783e0627891c3fb2b0fde8c42d703b24521f92c60bfef0b35078059711`, non è stata sovrascritta.

## 5. Inventario e test base

| nodo | CPU capability PyTorch | matmul/exp/logsigmoid/linear/softmax | pytest |
|---|---|---|---:|
| atlas5, Haswell AVX2 | `AVX2` | pass, risultati finiti | **41 passed in 3.28 s** |
| argo3, Ivy Bridge AVX | `DEFAULT` | pass, risultati finiti | **41 passed in 3.74 s** |

La capability `DEFAULT` su argo3 è coerente con l'assenza di AVX2. Il test Gloo sintetico su `eno1`, MTU 1500, ha trasferito byte-identico un FP32 `[1,1,3584]` da 14.336 byte. Su 100 round-trip persistenti: mediana 0,647 ms, min 0,527 ms, max 1,257 ms; shutdown pulito.

## 6. Stress `exp/logsigmoid`

Sono stati avviati 50 container/processi indipendenti nominati, ognuno con 20 ripetizioni del percorso `exp/logsigmoid`, quattro thread e capability registrata prima del test.

```text
processi pass/fail: 50/0
SIGILL:             0
tempo totale:       171 s
tempo/container:    circa 3,4 s, incluso startup
```

I container non sono stati lanciati con `--rm`, così un eventuale crash sarebbe rimasto ispezionabile. Nessun container è terminato con exit 132 e il journal kernel non contiene trap nuovi.

## 7. Microtest modello

Ogni famiglia è stata eseguita in un processo separato su argo3.

| test | exit | shape finale | finite | confronto numerico con MKL |
|---|---:|---|---|---|
| fake quantizzazione | 0 | `[4096,4096]` | sì | checksum identico `88.790863` |
| RMSNorm | 0 | `[8,3584]` | sì | identico `6.219595` |
| attention | 0 | `[1,28,8,128]` | sì | identico `6.911742` |
| GLA recurrent | 0 | `[1,28,8,128]` | sì | identico `5.288283` |
| GLA block real-size | 0 | `[1,1,3584]` | sì | delta checksum `4,8e-7` |
| attention block real-size | 0 | `[1,1,3584]` | sì | delta checksum `2,4e-7` |
| linear 3584→3584 | 0 | `[1,3584]` | sì | delta `2,4e-5` |
| linear 3584→18944 | 0 | `[1,18944]` | sì | delta `1,4e-4` |
| linear 18944→3584 | 0 | `[1,3584]` | sì | delta `9,2e-4` |
| lm_head sintetico | 0 | `[1,152064]` | sì | delta checksum `0,0011` |

Le piccole differenze delle riduzioni BLAS sono attese tra backend diversi. Shape, dtype e finitezza coincidono; non sono emerse divergenze funzionali nei test unitari.

## 8. Trenta ArgoStage reali indipendenti

Ogni processo ha montato il checkpoint originale read-only, caricato layer 14–27, norm e lm_head, eseguito un forward su hidden sintetico e terminato completamente.

```text
processi pass/fail: 30/0
SIGILL:             0
token locale:       1940 in 30/30 processi
forward min:        7128,400 ms
forward mediana:    7426,210 ms
forward media:      7436,170 ms
forward max:        8061,568 ms
byte logici:        15384997376
RSS/HWM esempio:    15150,88 MiB
swap processo:      0 MiB
```

Il precedente runtime MKL aveva una mediana ArgoStage di 4912,939 ms: OpenBLAS/MKL = **1,512×**. Non sono comparsi OOM, output non finiti o trap kernel.

## 9. AtlasStage

Tre processi indipendenti hanno caricato embedding e layer 0–13 dal subset read-only. Tutti hanno prodotto shape corretta, output finito, checksum identico `-138.773895`, cache a posizione 1 e swap zero.

Il primo forward freddo è durato 20.344,556 ms; i due successivi 7398,690 e 7235,621 ms, mediana calda **7317,155 ms**. Rispetto alla mediana Goal 6 di 5465,185 ms il rapporto è **1,339×**. RSS/HWM massimo del test locale: circa 13.145 MiB.

## 10. Prestazioni per operazione

| operazione argo3 | vecchio MKL | nuovo OpenBLAS | nuovo/vecchio |
|---|---:|---:|---:|
| GLA block decode | 341,327 ms | 491,500 ms | 1,440× |
| attention block decode | 378,126 ms | 576,570 ms | 1,525× |
| GLA gate QuantLinear | 130,638 ms | 166,762 ms | 1,276× |
| attention gate QuantLinear | 130,548 ms | 182,306 ms | 1,396× |
| GLA MLP | 276,652 ms | 370,192 ms | 1,338× |
| attention MLP | 276,064 ms | 393,414 ms | 1,425× |
| attention core | 37,027 ms | 72,348 ms | 1,954× |
| ArgoStage reale | 4912,939 ms | 7426,210 ms | 1,512× |
| lm_head reale, percorso Gloo | 109,251 ms | 486,344 ms | 4,451× |
| AtlasStage locale caldo | 5465,185 ms | 7317,155 ms | 1,339× |

La regressione maggiore è nel `lm_head`; non è stato tentato alcun tuning. La priorità di questo goal era eliminare il percorso ISA non sicuro.

## 11. Forward tecnico Gloo reale

Sono state eseguite 5 sessioni/processi distribuiti indipendenti con token tecnico 42, checkpoint read-only e rete host vincolata a `eno1`. Ogni sessione ha effettuato tre forward con cache ricreate coerentemente; in totale 15 forward reali. Nessun tokenizer o testo è stato usato.

| metrica, mediana sui 15 forward | valore |
|---|---:|
| AtlasStage | 8220,774 ms |
| chiamata send hidden | 0,116 ms |
| Argo layer 14–27 | 7486,404 ms |
| norm | 0,352 ms |
| lm_head | 486,344 ms |
| argmax | 0,455 ms |
| ritorno token | 0,159 ms |
| end-to-end | **16194,007 ms** |

Rispetto ai 10683,879 ms del Goal 7, il rapporto end-to-end è **1,516×**. Tutte le sessioni hanno prodotto token 16 in 15/15 forward, hidden di confine byte-identico, shutdown pulito e nessun nuovo process group per iterazione. Il token differisce dal 320 della build MKL: la cascata di piccole differenze numeriche del backend cambia l'argmax finale, ma il nuovo runtime è internamente deterministico.

Picchi osservati nelle sessioni distribuite:

- atlas5 HWM massimo: 13653,12 MiB;
- argo3 HWM massimo: 15660,75 MiB;
- swap processo su entrambi: 0 MiB.

## 12. SIGILL e integrità

Il controllo finale `journalctl -k` su argo3, effettuato dopo microtest, stress degli stage e sessioni Gloo, non contiene `invalid opcode`, `SIGILL` o trap in `libtorch_cpu.so` per questa campagna. Totale osservato con il nuovo runtime:

- 50 processi `exp/logsigmoid`: 0 SIGILL;
- 10 microtest separati: 0 SIGILL;
- 30 ArgoStage completi: 0 SIGILL;
- 3 AtlasStage completi: 0 SIGILL;
- 5 sessioni Gloo reali, 15 forward: 0 SIGILL;
- pytest su entrambi i nodi: 82 test complessivi passati;
- nessuna generazione testuale;
- checkpoint e partizione invariati e montati read-only;
- nessuna modifica a tokenizer, pesi o protocollo Gloo;
- nessun uso di CUDA, MKL, oneDNN, Triton, fla o FlashAttention.

## 13. Log grezzi

I log sono conservati in `goal86_logs/`, inclusi inventari runtime/ELF, pytest, stress `exp/logsigmoid`, microtest, 30 ArgoStage, tre AtlasStage, Gloo sintetico, cinque coppie di log Gloo reali e controlli kernel finali. I container diagnostici nominati sono stati mantenuti dove previsto per consentire ispezione successiva.

## Raccomandazione finale

**GO-RUNTIME-STABLE**. Usare `spikingbrain-cpu:goal8.6-openblas` come unica base del prossimo tentativo Goal 8, mantenendo il target Ivy Bridge e senza reintrodurre MKL/MKLDNN. La perdita prestazionale è accettata: circa 16,2 s per il forward tecnico distribuito contro 10,7 s con MKL. Prima di qualunque ottimizzazione, la nuova baseline deve restare questa immagine esatta e verificata.
