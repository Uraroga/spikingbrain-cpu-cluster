# Goal 8 — Prima generazione autoregressiva distribuita

## Decisione finale

**STOP — una seconda SIGILL su argo3 richiede diagnosi dell'ISA/runtime prima di proseguire.**

La prima generazione testuale reale è riuscita integralmente, ma la seconda sessione richiesta per la ripetibilità è terminata con `SIGILL`/exit 132 su argo3. In conformità alla guardia esplicita del Goal 8 non è stato effettuato alcun nuovo tentativo e non si procede al Goal 9.

## Immagine e preflight

È stata costruita una sola immagine `spikingbrain-cpu:goal8` su atlas5 e trasferita ad argo3 con `docker save | ssh ... docker load`. Image ID identico:

`sha256:81ac4c783e0627891c3fb2b0fde8c42d703b24521f92c60bfef0b35078059711`

Il preflight sulla stessa immagine ha superato:

- import di PyTorch 2.13.0+cpu;
- `cuda_available=false`;
- matmul FP32 64×64, output finito, exit code container 0 e nessun OOM;
- test Gloo sintetico persistente su `eno1` con exit pulito;
- 41 test del progetto.

Non si è verificata SIGILL nel preflight.

## Tokenizer offline

Il checkpoint dichiara `Qwen2Tokenizer`. È stato usato direttamente `transformers.Qwen2Tokenizer`, non `AutoModel` e non un modello Hugging Face automatico, con Transformers 4.55.4 e `local_files_only=True`. Il runtime aveva `HF_HUB_OFFLINE=1` e `TRANSFORMERS_OFFLINE=1`; nessun download è avvenuto durante test o generazione.

I file sono stati copiati tramite l'IP Ethernet privato in una directory separata e montata read-only:

`$HOME/Modelli/SpikingBrain/SpikingBrain-7B-W8ASpike-tokenizer`

| file | byte | SHA-256 sorgente/destinazione |
|---|---:|---|
| `tokenizer_config.json` | 7,258 | `c7462055715b3b866c9761ad749bfb364af26eb3776e07d20d14f98509e5ddc4` |
| `special_tokens_map.json` | 761 | `22bb30aee4ffc3a1ece0ccf70ee85207eb027658d08574d19df7d05ae3abdfdc` |
| `added_tokens.json` | 605 | `58b54bbe36fc752f79a24a271ef66a0a0830054b4dfad94bde757d851968060b` |
| `vocab.json` | 3,383,407 | `87a257b04b17642a0688c98cd1df89c398bda4fee532d6f88b38a659ecb4ac8d` |
| `merges.txt` | 1,671,853 | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |

Metadati verificati:

- vocab base: 151,643; lunghezza con added tokens: 151,665;
- BOS `<|im_start|>` = 151644, ma `add_bos_token=false`;
- EOS `<|endoftext|>` = 151643;
- PAD = EOS = 151643;
- round-trip: `Hello` → `[9707]` → `Hello`.

Non è stato inventato o aggiunto manualmente alcun token speciale.

## Protocollo generativo

Il protocollo Goal 7 resta disponibile ed è stato esteso con entry point separati. Atlas gestisce tokenizer, token list, embedding/layer 0–13 e testo. Argo gestisce layer 14–27, norm, proiezione della sola ultima posizione, greedy argmax e restituisce soltanto un `int64`.

Il prefill invia una lunghezza validata e un hidden FP32 `[1,L,3584]`. Nel decode invia soltanto `[1,1,3584]`. Un controllo continue/stop evita che un rank resti in attesa dopo EOS o `max_new_tokens`. Ogni process group è persistente per l'intera sessione.

## Sessione 1 — riuscita

**PROMPT:**

`Hello`

**TOKEN PROMPT:**

`[9707]` — lunghezza prefill 1, sotto il limite di 8.

**TOKEN GENERATI:**

`[11, 323, 10565]`

**TESTO COMPLETO:**

`Hello, and welcome`

**CONTINUAZIONE GENERATA:**

`, and welcome`

EOS 151643 non è comparso; arresto normale per `max_new_tokens=3`. Il testo è riportato senza correzioni o reinterpretazioni.

### Cache

Su ciascun rank sono presenti 14 cache, per un totale di 28 layer: sette GLA e sette KV per stage. Le cache partono vuote; lo stato restituito da ogni layer viene passato direttamente allo step successivo e non viene mai azzerato o ricreato semanticamente durante la sessione.

| evento | posizione su tutti i 28 layer | byte cache per rank |
|---|---:|---:|
| dopo prefill | 1 | 12,873,728 |
| dopo elaborazione token generato 1 | 2 | 12,902,400 |
| dopo elaborazione token generato 2 | 3 | 12,931,072 |

La crescita è esattamente 28,672 byte per nuovo token processato e deriva dalle sette cache KV; gli stati GLA hanno dimensione costante. Il terzo token viene restituito e decodificato ma, poiché termina la sessione, non viene nuovamente passato nel modello.

### Tempi

| operazione | tempo |
|---|---:|
| tokenizzazione | 0.331 ms |
| prefill AtlasStage | 5,077.381 ms |
| send hidden prefill | 0.597 ms |
| prefill ArgoStage layer | 5,959.496 ms |
| lm_head primo token | 114.663 ms |
| nuovo token 1, elapsed stimato | 11,154.705 ms |
| nuovo token 2, elapsed stimato | 9,844.696 ms |
| nuovo token 3, elapsed stimato | 9,847.147 ms |
| media per nuovo token | 10,282.794 ms |
| generazione complessiva | 30,848.382 ms |

Il costo Gloo rimane inferiore al millisecondo; il tempo è dominato dai due stage CPU.

### RAM

| nodo | RSS finale | HWM | MemFree | MemAvailable | swap processo |
|---|---:|---:|---:|---:|---:|
| atlas5 | 13,268.11 MiB | 13,268.11 MiB | 3,685.30 MiB | 28,526.50 MiB | 0 MiB |
| argo3 | 15,175.20 MiB | 15,175.20 MiB | 373.24 MiB | 30,214.70 MiB | 0 MiB |

Su argo RSS è passato da 15,161.10 a 15,175.20 MiB nei tre step; su atlas non è stata osservata crescita anomala. Il basso MemFree con MemAvailable elevato è coerente con page cache mmap recuperabile. Nessun processo ha usato swap.

La prima sessione ha riportato `clean_shutdown=true` su entrambi i rank.

## Sessione 2 — SIGILL e arresto

La seconda sessione è partita con processi e cache nuovi, come richiesto. Durante l'attesa del primo token atlas ha ricevuto:

`Gloo connection closed by peer [192.168.1.64]`

Il container argo è terminato senza traceback Python con exit code **132**, cioè SIGILL. Il kernel di argo ha registrato alle 19:41:55:

`trap invalid opcode ... in libtorch_cpu.so`

Questo costituisce una nuova SIGILL dopo quella non riproducibile del Goal 7. Non è stata tentata alcuna ripetizione automatica.

Diagnostica raccolta:

- CPU: Intel i3-3240, family 6 model 58 stepping 9, AVX/F16C ma non AVX2;
- PyTorch 2.13.0+cpu;
- kernel `7.0.0-28-generic`;
- image inspect completo nel file diagnostico;
- `journalctl -k` disponibile e contenente il trap in `libtorch_cpu.so`;
- `dmesg` non leggibile per permessi;
- `coredumpctl` non disponibile/non ha prodotto dati;
- il container era stato avviato con `--rm`, quindi dopo il crash non era più disponibile per `docker inspect`; l'exit 132 è stato acquisito dal client Docker/SSH;
- nessun listener 29500 e nessun processo generativo residuo dopo il crash.

La failure handling di rank 0 ha rilevato immediatamente la chiusura del peer ed è terminata con errore invece di restare appesa. La ripetibilità greedy non può essere dichiarata, anche se la prima sessione ha prodotto un risultato deterministico internamente.

## File prodotti

- `scripts/distributed_generate.py`
- `scripts/generate_atlas.py`
- `scripts/generate_argo.py`
- `goal8_session1_atlas.json`, `goal8_session1_argo.json`
- `goal8_session2_atlas.json` e relativi stderr
- `goal8_preflight_atlas.json`, `goal8_preflight_argo.json`
- `goal8_sigill_diagnostics.txt`
- `goal8_generation.json`

## Bloccante

Prima di qualunque nuova generazione occorre diagnosticare l'istruzione illegale in `libtorch_cpu.so`. Il passo successivo consigliato non è Goal 9, ma una sessione diagnostica controllata che conservi il container/core, identifichi l'offset/simbolo del crash e verifichi se PyTorch 2.13.0+cpu sta selezionando sporadicamente un kernel ISA non supportato dall'Ivy Bridge di argo3.
