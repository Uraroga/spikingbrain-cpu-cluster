# Goal 2.5 — Ambiente Docker e benchmark atlas5/argo3

## Decisione

**GO — possiamo iniziare il Goal 3: loader selettivo e stage sintetici.**

La stessa immagine Docker CPU-only è stata eseguita con successo su atlas5 e argo3. PyTorch è compatibile con l'i3-3240, tutti i 15 test passano sui due nodi e non sono emerse illegal instruction o dipendenze GPU. Argo3 è però sensibilmente più lento negli operatori elementwise e soprattutto in QuantLinear; il Goal 3 dovrà misurare proiezioni e blocchi sintetici prima di considerare definitivo il taglio 16/12.

## Ambiente Docker

File creati:

- `Dockerfile`: immagine Python slim amd64, PyTorch CPU, pytest, codice e benchmark incorporati, runtime come utente non-root `app` (UID/GID 10001);
- `requirements.lock`: versioni dirette e transitive bloccate;
- `.dockerignore`: esclude `.venv`, `.git`, cache Python/pytest, temporanei, shard Safetensors, GGUF e directory tipiche di modelli/checkpoint.

Base image:

```text
python:3.12.3-slim-bookworm
sha256:fd3817f3a855f6c2ada16ac9468e5ee93e361005bd226fd5a5ee1a504e038c84
```

Versioni principali:

| Componente | Versione |
|---|---:|
| Python | 3.12.3 |
| PyTorch | 2.13.0+cpu |
| pytest | 9.1.1 |
| Base OS container | Debian Bookworm slim |
| CUDA | assente/non disponibile |

Non sono installati CUDA, Triton, `fla`, FlashAttention o NumPy. L'assenza opzionale di NumPy produce un warning PyTorch durante l'avvio dei test, ma il progetto non usa NumPy e i risultati non ne sono influenzati.

Build effettuata una sola volta su atlas5:

```bash
docker build --pull --tag spikingbrain-cpu:goal2.5 .
```

L'immagine occupa 290.036.657 byte e usa `app`, non root, a runtime. Poiché il codice è incorporato nell'immagine e non è stato montato il workspace, i container non hanno creato file root-owned nel progetto.

## Identità dell'immagine

Su entrambi i nodi:

```text
Image ID:    sha256:c22fbd73a0b16601364d8444d1c410f3cb06ddbce970ae5ab9e6698f8ee0e552
RepoDigest:  spikingbrain-cpu@sha256:c22fbd73a0b16601364d8444d1c410f3cb06ddbce970ae5ab9e6698f8ee0e552
```

ID e digest identici dimostrano che su argo3 è stata caricata la build di atlas5, non una seconda ricostruzione.

## Test atlas5

Esecuzione dentro il container:

```bash
docker run --rm spikingbrain-cpu:goal2.5 python -m pytest -q
```

Risultato:

```text
15 passed, 1 warning in 1.56s
```

Non è stato necessario modificare alcun algoritmo rispetto al Goal 2.

## Benchmark atlas5 in Docker

Comando comune ai due nodi:

```bash
docker run --rm spikingbrain-cpu:goal2.5 \
  python scripts/benchmark_cpu_ops.py --iterations 20 --warmup 5 --threads 4
```

Ambiente rilevato:

| Campo | Valore |
|---|---|
| CPU | Intel Core i5-4590 @ 3.30 GHz |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cpu |
| Thread PyTorch | 4 |
| RAM disponibile | 29.262,9 MiB |
| `cuda_available` | false |

Mediane su 20 iterazioni dopo 5 warmup, sempre in `torch.inference_mode()`:

| Operazione | Mediana | Min–max |
|---|---:|---:|
| RMSNorm `[1,32,3584]` | 0,084 ms | 0,081–0,106 ms |
| SwiGLU core `[1,8,18944]` | 0,075 ms | 0,073–0,081 ms |
| attention prefill L64, H28/KV4, D128 | 0,966 ms | 0,813–14,209 ms |
| attention decode query 1/cache 4096 | 36,233 ms | 26,840–59,229 ms |
| GLA prefill L32, H28, D128 | 36,421 ms | 27,447–65,920 ms |
| GLA decode 1 token con cache | 1,368 ms | 1,035–3,391 ms |
| QuantLinear allocante 1024×1024 | 2,108 ms | 1,867–7,529 ms |
| QuantLinear con buffer 1024×1024 | 0,542 ms | 0,527–0,578 ms |

### Confronto con la precedente baseline `.venv`

| Operazione | Baseline Goal 2 | Docker | Docker/baseline |
|---|---:|---:|---:|
| RMSNorm | 0,073 | 0,084 | 1,15× |
| SwiGLU | 0,074 | 0,075 | 1,01× |
| attention prefill | 1,618 | 0,966 | 0,60× |
| attention decode | 33,652 | 36,233 | 1,08× |
| GLA prefill | 37,985 | 36,421 | 0,96× |
| GLA decode | 1,786 | 1,368 | 0,77× |
| QuantLinear allocante | 3,041 | 2,108 | 0,69× |
| QuantLinear buffer | 0,613 | 0,542 | 0,88× |

Non c'è una regressione sistematica. Le differenze negative maggiori sono +15% su RMSNorm, pari a soli 0,011 ms, e +8% su attention decode; più operatori risultano invece più rapidi. I test non fissano affinity, governor o carico dell'host e i min/max mostrano jitter, quindi queste variazioni sono compatibili con un microbenchmark CPU non isolato. Non è stata individuata una differenza funzionale tra gli ambienti.

## Rimozione di `.venv`

Dopo il successo sia dei test sia del benchmark Docker su atlas5, `.venv` (898 MiB) è stata rimossa dal progetto tramite spostamento nel cestino, quindi è ancora recuperabile finché il cestino non viene svuotato. Rimane esclusa da `.gitignore` e `.dockerignore`.

Da questo punto i comandi del progetto usano esclusivamente l'immagine Docker; nessun Python/PyTorch host è richiesto.

## Trasferimento ad argo3

È stata trasferita direttamente l'immagine locale sulla rete Ethernet privata, senza registry e senza ricostruzione:

```bash
docker save spikingbrain-cpu:goal2.5 |
  ssh <USER>@192.168.1.64 docker load
```

Sono transitati soltanto i layer dell'immagine Docker. Non sono stati copiati file del checkpoint o pesi.

## Compatibilità PyTorch/ISA su argo3

Prima dei test completi è stato eseguito un container minimale con import PyTorch e moltiplicazione tensoriale:

```text
torch 2.13.0+cpu
tensor [3.0, 6.0]
cuda_available False
```

Il test passa senza illegal instruction. La wheel manylinux CPU funziona quindi sull'Ivy Bridge i3-3240/AVX. PyTorch sceglieva 2 thread per default sul nodo, ma il benchmark li imposta esplicitamente a 4 come richiesto.

## Test argo3

Risultato nello stesso container:

```text
15 passed, 1 warning in 1.79s
```

Non sono stati installati Python, PyTorch o pytest sull'host argo3.

## Benchmark argo3

Ambiente rilevato:

| Campo | Valore |
|---|---|
| CPU | Intel Core i3-3240 @ 3.40 GHz |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cpu |
| Thread PyTorch | 4 |
| RAM disponibile | 30.524,0 MiB |
| `cuda_available` | false |

| Operazione | Mediana | Min–max |
|---|---:|---:|
| RMSNorm `[1,32,3584]` | 0,219 ms | 0,216–0,239 ms |
| SwiGLU core `[1,8,18944]` | 0,453 ms | 0,448–0,466 ms |
| attention prefill L64, H28/KV4, D128 | 3,589 ms | 3,059–3,762 ms |
| attention decode query 1/cache 4096 | 37,304 ms | 37,072–38,527 ms |
| GLA prefill L32, H28, D128 | 43,287 ms | 42,923–49,108 ms |
| GLA decode 1 token con cache | 1,793 ms | 1,775–1,815 ms |
| QuantLinear allocante 1024×1024 | 5,323 ms | 5,208–7,487 ms |
| QuantLinear con buffer 1024×1024 | 2,207 ms | 2,195–2,237 ms |

## Confronto diretto

Rapporto maggiore di 1 significa che argo3 impiega più tempo.

| Operazione | atlas5 | argo3 | argo3/atlas5 |
|---|---:|---:|---:|
| RMSNorm | 0,084 ms | 0,219 ms | **2,61×** |
| SwiGLU core | 0,075 ms | 0,453 ms | **6,04×** |
| attention prefill L64 | 0,966 ms | 3,589 ms | **3,72×** |
| attention decode cache 4096 | 36,233 ms | 37,304 ms | **1,03×** |
| GLA prefill L32 | 36,421 ms | 43,287 ms | **1,19×** |
| GLA decode cached | 1,368 ms | 1,793 ms | **1,31×** |
| QuantLinear allocante | 2,108 ms | 5,323 ms | **2,53×** |
| QuantLinear con buffer | 0,542 ms | 2,207 ms | **4,07×** |

I kernel grandi e memory-bound, come attention decode su cache 4096, sono quasi equivalenti. Le operazioni piccole/vectorized e QuantLinear penalizzano invece molto argo3. Le cause plausibili sono due core fisici con Hyper-Threading contro quattro core fisici e AVX contro AVX2/FMA; il benchmark non isola singolarmente ISA, numero di core e banda memoria, quindi non attribuisce quantitativamente il rallentamento a una sola causa.

Per la partizione prevista, i dati più importanti sono GLA decode 1,31×, attention decode 1,03× e QuantLinear 2,53–4,07×. Poiché ogni layer contiene molte proiezioni QuantLinear, argo3 potrebbe diventare il collo di bottiglia anche se i soli core attention/GLA appaiono accettabili. Questo non giustifica uno STOP prima del loader, ma rende obbligatorio un benchmark di blocco completo sintetico e potrà motivare uno spostamento di ulteriori layer verso atlas5.

## Vincoli rispettati

- Nessuno shard Safetensors è stato aperto o caricato.
- Nessun peso è stato copiato su atlas5 o inserito nell'immagine.
- Non sono stati usati `from_pretrained()`, modello completo, AtlasStage, ArgoStage o Gloo.
- Il checkpoint originale su argo3 non è stato montato nel container né modificato.
- `$HOME/Modelli/SpikingBrain` non è stata creata su atlas5.
- L'immagine contiene esclusivamente codice Goal 2, PyTorch CPU, pytest e dipendenze Python bloccate.

## Raccomandazione finale

Procedere al Goal 3 con la stessa immagine come base, mantenendo inizialmente pesi e stage sintetici. Prima di caricare pesi reali occorre aggiungere un benchmark rappresentativo delle proiezioni 3584/18944 e di un blocco completo su entrambi i nodi. Se quel benchmark confermerà una penalità QuantLinear vicina a 4× su argo3, sarà opportuno riesaminare il taglio layer 16/12 o la fake-quantizzazione per-forward prima dell'inferenza distribuita.
