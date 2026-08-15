# Goal 8.5 — Diagnosi SIGILL su argo3

## Decisione

**GO-RUNTIME — il crash è localizzato, ma nessuna delle limitazioni ISA disponibili rende stabile questa build. Serve una build/versione PyTorch CPU diversa e validata per Ivy Bridge prima di ripetere Goal 8.**

Non sono state eseguite generazioni, Gloo con pesi, modifiche ai pesi o cambi di partizione. Goal 9 non è stato avviato.

## Risultato principale

Tutti i trap kernel disponibili, inclusi i due nuovi crash diagnostici, puntano allo stesso offset:

`libtorch_cpu.so + 0xb4655eb`

`addr2line` lo risolve come:

`mkl_vml_kernel_sExp_Z0HAynn`

L'istruzione all'indirizzo è:

`vmovups ...,%zmm12`

È un'istruzione AVX-512 su registro ZMM. L'Intel i3-3240 Ivy Bridge supporta AVX/F16C ma non AVX2 né AVX-512. La SIGILL è quindi spiegata con precisione: il dispatcher Intel MKL VML entra intermittentemente nel kernel AVX-512 di `sExp` su una CPU che non lo supporta.

## Inventario runtime

Eseguito dentro l'immagine Goal 8 originale, senza variabili ISA impostate:

| proprietà | valore |
|---|---|
| PyTorch | 2.13.0+cpu |
| capability PyTorch | `DEFAULT` |
| MKL disponibile | sì |
| oneDNN/MKLDNN disponibile | sì |
| thread intra-op predefiniti | 2 |
| thread inter-op predefiniti | 2 |
| variabili ATEN/MKL/MKLDNN/DNNL/OMP | nessuna |

I test del modello usano esplicitamente quattro thread.

Configurazione build rilevante:

- GCC 13.3, C++20, Release;
- Intel oneAPI MKL 2024.2, build 20240605;
- Intel MKL-DNN/oneDNN 3.12.0, commit `80afa710…`;
- OpenMP 4.5;
- `USE_MKL=ON`, `USE_MKLDNN=ON`, `USE_OPENMP=ON`;
- `PERF_WITH_AVX=1`, `PERF_WITH_AVX2=1`;
- `CPU capability usage: DEFAULT`.

## CPU reale

Argo3 monta un Intel Core i3-3240, family 6, model 58, stepping 9, microcode `0x21`, due core/quattro thread. Le flags complete sono conservate nei log. Quelle ISA rilevanti sono:

`sse sse2 ssse3 sse4_1 sse4_2 popcnt pclmulqdq xsave avx f16c xsaveopt`

Sono assenti `avx2`, `avx512*`, FMA3 e qualunque estensione ZMM. `torch.backends.cpu.get_cpu_capability()` restituisce quindi plausibilmente `DEFAULT`; il problema non è un valore ATen che dichiara AVX2, ma un dispatch interno a MKL VML.

## Identificazione ELF

Path caricato:

`/usr/local/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so`

| proprietà | valore |
|---|---|
| dimensione | 434,184,800 byte |
| SHA-256 | `ae0f4bc33ffe73f4eb85b2fd03b036c68cf5ab6139995f6a2345f5962c1bbb81` |
| build-id | `8ec08ec8f71de04ee2baa46c0dbe262858b1e27c` |
| ELF | x86-64, DYN, GNU/Linux |
| simboli | non stripped |

`ldd` mostra `libc10`, glibc, libstdc++ e `libgomp.so.1`; non mostra librerie MKL/oneDNN separate. I loro kernel e simboli sono incorporati in `libtorch_cpu.so`; OpenMP è collegato dinamicamente tramite la copia `libgomp` inclusa in PyTorch.

## Trap kernel e simbolizzazione

Il journal contiene tre eventi antecedenti alla diagnosi e due riprodotti durante lo stress conservativo:

| timestamp argo3 | PID | RIP | base mapping | offset | error |
|---|---:|---|---|---|---:|
| 17:26:02 | 7880 | `0x7d30394f75eb` | `0x7d302e092000` | `0xb4655eb` | 0 |
| 19:26:49 | 11291 | `0x7e20f16f75eb` | `0x7e20e6292000` | `0xb4655eb` | 0 |
| 19:41:55 | 12684 | `0x74f0886f75eb` | `0x74f07d292000` | `0xb4655eb` | 0 |
| 19:59:10 | 23200 | `0x7ed063cf75eb` | `0x7ed058892000` | `0xb4655eb` | 0 |
| 19:59:46 | 23737 | `0x7cc71ccf75eb` | `0x7cc711892000` | `0xb4655eb` | 0 |

L'offset è identico nonostante ASLR. `addr2line -Cfipe`, `objdump -d -C` e la symbol table non-stripped identificano il kernel `mkl_vml_kernel_sExp_Z0HAynn+0xab`. La sequenza a `0xb4655eb` usa EVEX/ZMM e i dati sono etichettati `vsexp_data_avx512`.

Il journal non riporta il numero di core. `dmesg` diretto non è consentito all'utente; `coredumpctl` non è disponibile e non ci sono core. I container diagnostici falliti sono invece conservati con metadata e log completi.

## Microtest sintetici baseline

Ogni caso è stato eseguito in un processo/container distinto, con timeout e container conservato. Tre ripetizioni interne salvo lo stress successivo.

| caso | risultato |
|---|---|
| `torch.exp`, 1,000,003 FP32 | pass |
| `F.logsigmoid`, 1,000,003 FP32 | pass |
| `F.linear` 64→64 | pass |
| `F.linear` 3584→3584 | pass |
| `F.linear` 3584→18944 | pass |
| `F.linear` 18944→3584 | pass |
| lm_head 3584→152064 | pass; 53.19 s inclusa inizializzazione casuale di ~2.03 GiB |
| fake quant div/round/clamp/mul | pass |
| RMSNorm 3584 | pass |
| softmax/matmul attention | pass |
| GLA recurrent | pass |
| singolo GLA block reale-size sintetico | pass |
| singolo attention block reale-size sintetico | pass |

Il primo tentativo lm_head aveva exit 1 per un errore nel dispatch del microtest (`ValueError: lm_head`), non SIGILL né OOM; la versione corretta, conservata come `goal85-base-lm_head-v2`, è passata.

Nessun singolo microtest breve ha riprodotto la SIGILL. Tuttavia la simbolizzazione rende `exp`/`logsigmoid` la famiglia coinvolta; nel modello il percorso più evidente è `F.logsigmoid` in GLA.

## Stress `logsigmoid`

Sono stati eseguiti 10 processi indipendenti per modalità, ciascuno con 100 chiamate su 1,000,003 FP32: 5,000 chiamate complessive in 50 processi.

| modalità | processi pass/fail | mediana interna | min/max | variazione mediana |
|---|---:|---:|---:|---:|
| baseline | 10/0 | 851.820 ms | 848.137/871.369 | riferimento |
| `ATEN_CPU_CAPABILITY=default` | 10/0 | 848.841 ms | 843.275/857.438 | -0.35% |
| `DNNL_MAX_CPU_ISA=AVX` | 10/0 | 864.465 ms | 849.149/867.034 | +1.48% |
| `MKL_ENABLE_INSTRUCTIONS=AVX` | 10/0 | 850.625 ms | 846.430/863.292 | -0.14% |
| ATen default + DNNL AVX | 10/0 | 856.483 ms | 846.827/870.306 | +0.55% |

Tutti riportano capability `DEFAULT` e le variabili attese nell'ambiente. L'assenza di crash in questo test non è sufficiente: il difetto è per-processo e raro.

## Test delle limitazioni ISA

### `ATEN_CPU_CAPABILITY=default`

Non cambia la capability: la baseline era già `DEFAULT`. Non controlla il kernel MKL VML incriminato. Tre ArgoStage reali indipendenti sono passati, mediana 4,917.859 ms contro 4,912.939 ms baseline (+0.10%), ma questo non dimostra protezione.

### `DNNL_MAX_CPU_ISA=AVX`

Limita oneDNN, mentre il simbolo del crash appartiene a MKL VML. Tre ArgoStage reali indipendenti sono passati, mediana 4,918.055 ms (+0.10%), ma non c'è evidenza che questa variabile raggiunga `sExp`.

### ATen default + DNNL AVX

Tre ArgoStage indipendenti sono passati, mediana 4,913.192 ms (+0.01%). Anche la combinazione non controlla MKL VML.

### `MKL_ENABLE_INSTRUCTIONS=AVX`

È stata valutata solo dopo aver identificato MKL VML come sorgente. MKL conferma di leggere la variabile, stampando che il supporto AVX-only è deprecato e che userà SSE4.2. Tre primi ArgoStage sono passati, mediana 4,933.362 ms (+0.42%).

La configurazione finale più conservativa ha combinato:

```text
ATEN_CPU_CAPABILITY=default
DNNL_MAX_CPU_ISA=AVX
MKL_ENABLE_INSTRUCTIONS=AVX
```

In dieci nuovi processi ArgoStage reali, **due sono terminati SIGILL** (run 4 e 9, exit 132, nessun OOM). Entrambi i trap sono ancora `libtorch_cpu.so+0xb4655eb`, `mkl_vml_kernel_sExp_Z0HAynn`, nonostante il warning MKL SSE4.2. I restanti otto hanno mediana 4,916.553 ms, ma il tasso osservato 2/10 rende la configurazione inutilizzabile.

Questo dimostra che `MKL_ENABLE_INSTRUCTIONS=AVX` viene letto ma non impedisce a questo specifico percorso VML di raggiungere il kernel AVX-512 nella build 2024.2 incorporata. Nessuna delle variabili testate è una mitigazione stabile.

## ArgoStage in processi indipendenti

Prima dello stress finale sono stati eseguiti tre processi per ciascuna delle cinque modalità, tutti completi e numericamente coerenti (token tecnico locale 1940):

| modalità | processi | mediana forward | min/max |
|---|---:|---:|---:|
| baseline | 3 | 4,912.939 ms | 4,910.560/8,125.664 |
| ATen default | 3 | 4,917.859 ms | 4,912.183/4,928.926 |
| DNNL AVX | 3 | 4,918.055 ms | 4,917.324/4,931.921 |
| MKL AVX | 3 | 4,933.362 ms | 4,918.980/4,936.740 |
| ATen default + DNNL AVX | 3 | 4,913.192 ms | 4,910.806/4,917.598 |

Lo stress finale porta il totale dei processi ArgoStage diagnostici a 25; due hanno riprodotto SIGILL. Nessun test ha usato Gloo, tokenizer o generazione.

## Conclusione tecnica

La SIGILL non è più “unknown”: è un dispatch errato/race interno al kernel MKL VML `sExp` incorporato in `libtorch_cpu.so` di PyTorch 2.13.0+cpu, build MKL 2024.2. ATen rileva correttamente `DEFAULT`; oneDNN non è il componente che esegue l'istruzione illegale. Anche la limitazione MKL documentabile dall'ambiente non rende stabile VML.

Per questo la decisione è **GO-RUNTIME**, non GO-ISA e non STOP-UNKNOWN. Il prossimo lavoro sicuro è costruire o selezionare un runtime PyTorch CPU differente, compatibile con Ivy Bridge e con MKL/VML diverso o disabilitato, quindi ripetere questi stessi microtest e almeno 20–50 ArgoStage indipendenti prima di qualunque generazione. Non è autorizzato riprendere Goal 8 con l'immagine attuale.

## Log e artefatti

- `goal85_inventory_raw.txt`: config PyTorch, capability, lscpu e cpuinfo completi.
- `goal85_elf_raw.txt`: file/hash/build-id/readelf/ldd e trap storici.
- `goal85_offset_raw.txt`: addr2line e disassembly dell'istruzione.
- `goal85_micro_baseline_summary.txt`: exit/timestamp microtest.
- `goal85_stress_summary.txt`: 50 processi `logsigmoid`.
- `goal85_argo_stage_summary.txt`: primi 15 ArgoStage indipendenti.
- `goal85_argo_conservative_summary.txt`: dieci ArgoStage conservativi.
- `goal85_conservative_failures.txt`: inspect/log/journal dei due container falliti.
- `goal85_logs/`: stdout/stderr grezzi dei microtest e degli stage.
- `$HOME/goal85_libtorch_cpu.so` su argo3: copia ELF usata per la simbolizzazione.

I container `goal85-*`, inclusi `goal85-argo-conservative-4` e `goal85-argo-conservative-9`, sono stati deliberatamente conservati su argo3 e non cancellati.
