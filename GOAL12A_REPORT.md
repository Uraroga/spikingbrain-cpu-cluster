# Goal 12A — Gloo controlled peer-loss test

## Esito

**CLEAN FAILURE (peer-loss rilevato senza hang e senza attendere il timeout Gloo).**

Atlas5/rank 0 ha rilevato la perdita di argo3/rank 1 alla prima nuova
operazione Gloo eseguita dopo il termine del proprio calcolo locale. L'errore è
stato `Connection closed by peer` durante il `dist.send()` dell'header del
successivo hidden autoregressivo. Rank 0 è terminato autonomamente con exit 1;
non è stato interrotto manualmente. Non sono rimasti processi, listener o
container in esecuzione.

## Vincoli rispettati

- Nessuna modifica al codice o al repository; worktree finale pulito.
- Nessun commit, push o aggiornamento del repository pubblico.
- Spegnimento eseguito manualmente da Sergio con `sudo init 0` solo dopo la
  prova dell'avvio autoregressivo e di almeno un token riuscito.
- Nessun `kill`, fault injection o riavvio automatico usato per provocare il
  guasto.
- Runtime esclusivo `spikingbrain-cpu:goal8.6-openblas`.
- Checkpoint, tokenizer, `src/` e `scripts/` montati read-only.
- Ethernet privata `eno1`, MTU 1500, Docker host networking e Gloo CPU.

## Implementazione osservata e semantica di `--timeout`

Commit ispezionato: `eba802a` (`eba802a...`, hash completo nel preflight).

`scripts/distributed_generate.py` espone `--timeout` in secondi (default 180)
e lo passa a `initialize()`. `protocol.initialize()` lo converte in
`timedelta(seconds=timeout_seconds)` e lo assegna a
`dist.init_process_group("gloo", ...)`. Le `send`, `recv` e la `barrier` finale
non impostano timeout individuali: usano il timeout del process group.

`shutdown()` esegue prima `dist.barrier()` e garantisce poi
`dist.destroy_process_group()` in un `finally`.

## Configurazione del test valido (attempt 3)

| Voce | Valore |
|---|---|
| Data | 2026-08-16, Europe/Rome (CEST, UTC+02:00) |
| Rank 0 | atlas5, 192.168.1.128, Intel i5-4590 |
| Rank 1 | argo3, 192.168.1.64, Intel i3-3240 |
| Interfaccia | `eno1`, Ethernet, MTU 1500 |
| Backend | Gloo, world size 2, porta 29612 |
| Docker | `--network host` |
| Immagine | `spikingbrain-cpu:goal8.6-openblas` |
| Image ID | `sha256:696d98683cee6eef018a73e5c70b402f9103fc24b230356975a7ef1685aed836` su entrambi |
| PyTorch | `2.13.0+openblas.ivybridge`, MKL/MKLDNN/CUDA disabilitati |
| Partizione | atlas5 embedding + layer 0–13; argo3 layer 14–27 + norm + `lm_head` |
| Thread | 4 per rank |
| Timeout | 75 s |
| `max_prompt_tokens` | 64 |
| `max_new_tokens` | 20 |
| `max_bytes` | 16,000,000,000 per rank |
| Decoding | greedy argmax |
| Prompt tokenizzati | 31 |

Prompt esatto:

> Explain in detail how two old computers can cooperate to run a large language model, describing the flow of data between the machines and the main technical limitations.

### Perché 75 secondi invece dei 30 preferiti

Il tentativo con 30 s ha misurato un limite di startup: rank 1 completa il
proprio stage mmap in circa 0,6 s e attende subito il primo hidden, mentre rank
0 deve materializzare il proprio stage e completare il prefill. Il `recv` di
rank 1 è scattato dopo 30,000 s prima che rank 0 potesse inviare il prefill.
Questo non era un peer-loss e non è contato come test valido. Sono stati scelti
75 s, il minimo prudente rispetto al percorso osservato, ancora molto inferiore
ai 180 s storici. Nel fault reale il timeout da 75 s non è stato raggiunto.

## Comandi esatti del test valido

Rank 1, avviato per primo via SSH su argo3:

```bash
docker run --name goal12a-argo-rank1 --network host \
  -e GLOO_SOCKET_IFNAME=eno1 \
  -v /home/sergio/Modelli/SpikingBrain/SpikingBrain-7B-W8ASpike:/model:ro \
  -v /home/sergio/goal12a-20260816/runtime/src:/app/src:ro \
  -v /home/sergio/goal12a-20260816/runtime/scripts:/app/scripts:ro \
  spikingbrain-cpu:goal8.6-openblas \
  python scripts/generate_argo.py \
  --master-addr 192.168.1.128 --port 29612 --peer 192.168.1.128 \
  --model-dir /model --max-bytes 16000000000 \
  --max-prompt-tokens 64 --max-new-tokens 20 --threads 4 --timeout 75
```

Rank 0, avviato su atlas5:

```bash
docker run --name goal12a-atlas-rank0 --network host \
  -e GLOO_SOCKET_IFNAME=eno1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v /home/sergio/Modelli/SpikingBrain/SpikingBrain-7B-W8ASpike-atlas:/model:ro \
  -v /home/sergio/Modelli/SpikingBrain/SpikingBrain-7B-W8ASpike-tokenizer:/tokenizer:ro \
  -v /home/sergio/Progetti/spikingbrain-cluster/src:/app/src:ro \
  -v /home/sergio/Progetti/spikingbrain-cluster/scripts:/app/scripts:ro \
  spikingbrain-cpu:goal8.6-openblas \
  python scripts/generate_atlas.py \
  --master-addr 192.168.1.128 --port 29612 --peer 192.168.1.64 \
  --model-dir /model --tokenizer-dir /tokenizer \
  --prompt 'Explain in detail how two old computers can cooperate to run a large language model, describing the flow of data between the machines and the main technical limitations.' \
  --max-bytes 16000000000 --max-prompt-tokens 64 \
  --max-new-tokens 20 --threads 4 --timeout 75
```

## Cronologia

| Timestamp CEST | Evento osservato |
|---|---|
| 14:57:20.372 | Container rank 1 avviato |
| 14:57:21.909 | Container rank 0 avviato |
| 14:57:38 circa | Stage rank 0 materializzato; prefill Atlas completato e rank 1 avvia il forward reale |
| 14:58:48–14:58:53 | Token 1 ricevuto da rank 0 e primo decode autoregressivo Atlas completato |
| 14:58:54–14:59:03 | Rank 1 elabora lo step seguente; token 2 completato |
| 14:59:04–14:59:09 | Rank 0 elabora token 2 e invia il successivo hidden |
| 14:59:10 | Rank 1 inizia il calcolo del token 3 |
| 14:59:11 | Sergio esegue `sudo init 0`; systemd registra `The system will power off now!` |
| 14:59:11.257 | Docker riceve SIGTERM come parte dello shutdown host |
| 14:59:11.686 | Primo campione SSH: argo3 irraggiungibile |
| 14:59:18 circa | Rank 0 inizia a elaborare token 3, già consegnato durante la finestra di shutdown |
| 14:59:21.333 | Docker: container non uscito entro 10 s da SIGTERM, applicazione della terminazione forzata |
| 14:59:21.552 | Rank 1 termina con exit 137, `OOMKilled=false` |
| 14:59:24.164 | Rank 0 emette JSON di errore `Connection closed by peer` |
| 14:59:24.944 | Container rank 0 terminato, exit 1 |
| 14:59:22 | Fine boot precedente / system power off |
| 15:00:41 | Boot corrente di argo3 iniziato |

## Punto del fault e token completati

Al momento dell'avvio dello spegnimento erano **sicuramente completati due
token** e rank 1 stava calcolando il terzo. Poiché Docker ha atteso 10 secondi
dopo SIGTERM prima della terminazione forzata, il terzo token è stato prodotto e
consegnato durante la sequenza di shutdown; rank 0 lo ha ricevuto ed elaborato.

Pertanto:

- token completamente chiusi prima del comando di fault: **2**;
- token 3: in corso all'inizio del fault, ma prodotto/ricevuto prima della
  terminazione effettiva del processo rank 1;
- il failure avviene inviando l'header dell'hidden derivato dal token 3.

Gli ID/testi dei token non sono recuperabili dal rank 1: l'implementazione li
serializza soltanto nel JSON finale, che non è stato emesso perché lo shutdown
host ha terminato il processo. Il conteggio è provato dalla sequenza alternata
dei contatori CPU timestampati e dal punto del traceback rank 0.

## Comportamento di atlas5 e latenza peer-loss

Eccezione esatta:

```text
RuntimeError: [/src/third_party/gloo/gloo/transport/tcp/pair.cc:553]
Connection closed by peer [192.168.1.64]:62885.
```

Stack applicativo:

```text
atlas() -> send_hidden(next_hidden, 1) -> dist.send(header, peer)
       -> isend(...) -> ProcessGroupGloo.send(...)
```

Misure di latenza:

- da inizio shutdown Docker (`14:59:11.257`) a eccezione rank 0
  (`14:59:24.164`): **12,907 s**;
- da primo campione di argo3 irraggiungibile (`14:59:11.686`) a eccezione:
  **12,477 s**;
- dalla terminazione effettiva del processo rank 1 (`14:59:21.552`) alla
  eccezione: **2,612 s**.

La maggior parte della prima durata non è attesa bloccante Gloo: rank 0 stava
elaborando localmente il token 3 già ricevuto. Il `send` successivo ha rilevato
la connessione chiusa rapidamente; non ha atteso 75 s.

## Timeout

- 30 s: insufficiente per il percorso iniziale asimmetrico; timeout normale su
  `recv` di rank 1 prima del prefill. Tentativo non valido come peer-loss.
- 75 s nel test valido: **non scattato**.
- Peer loss: segnalato da chiusura TCP esplicita (`Connection closed by peer`),
  non da `Timed out waiting ...`.

## `shutdown()` e barrier

Dopo l'eccezione applicativa, il driver stampa il JSON e nel `finally` chiama
`shutdown()`, cioè `barrier()` seguita comunque da `destroy_process_group()`.
Il JSON d'errore è timestampato `14:59:24.1637`; il traceback del re-raise
inizia a `14:59:24.1680`, circa 4 ms dopo. Il container termina alle
`14:59:24.9436`.

Quindi la barrier finale **non è rimasta bloccata e non ha atteso il timeout**;
con il peer assente è rientrata/fallita immediatamente e il process group è
stato distrutto. L'esatto testo di `shutdown_error` non è osservabile: il codice
aggiunge quel campo al report dopo che il JSON d'errore è già stato stampato e,
nel percorso di re-raise, non stampa nuovamente il report. Questa è una
limitazione della telemetria corrente, non una modifica richiesta o effettuata.

## Cleanup

### Atlas5

- rank 0: exit code **1**, `OOMKilled=false`;
- nessun processo `distributed_generate`/`generate_atlas` residuo;
- nessun listener sulla porta 29612;
- nessun container Goal 12A in esecuzione;
- container fermo conservato intenzionalmente per ispezione: `Exited (1)`;
- swap host 0.

### Argo3 dopo il riavvio

- rank 1: exit code Docker **137**, `OOMKilled=false`;
- restart policy `no`, quindi nessun riavvio automatico del workload;
- nessun processo generativo residuo;
- nessun listener sulla porta 29612;
- nessun container Goal 12A in esecuzione;
- container fermo conservato intenzionalmente per ispezione: `Exited (137)`;
- swap host 0.

Exit 137 non indica OOM in questo caso: il journal prova che Docker, durante lo
shutdown ordinato dell'host, ha inviato SIGTERM, atteso 10 s e poi forzato la
terminazione del container prima di completare il poweroff.

## Kernel e distinzione delle cause

Nel boot precedente di argo3:

- `SIGILL`: 0;
- `invalid opcode`: 0;
- `segfault`: 0;
- trap o riferimenti a `libtorch_cpu.so`: 0;
- OOM kill / `Killed process`: 0.

La sola riga contenente “OOM” è l'avvio ordinario del socket di
`systemd-oomd`, non un evento di memoria. Il journal registra esplicitamente lo
spegnimento richiesto dall'utente e la sequenza systemd/Docker; non esiste
evidenza di crash nativo non correlato.

## Artefatti

Directory principale: `/home/sergio/goal12a-20260816/`

- `preflight_atlas.txt`, `preflight_argo.txt`, tokenizzazione prompt;
- `attempt2/`: prova del limite startup con timeout 30 s;
- `attempt3/rank0.stdout`, `rank0.stderr`, exit/timestamp separati;
- `attempt3/rank1.stdout`, `rank1.stderr`, stato Docker post-boot;
- `attempt3/live_monitor.tsv`: RSS/CPU timestampati di entrambi i rank;
- log Docker timestampati;
- journal completo e kernel del boot precedente argo3;
- controlli cleanup su entrambi i nodi;
- hash dei sorgenti montati sui due host.

Il primo tentativo, anch'esso conservato, fallì prima del modello perché il
root filesystem del container era stato impostato read-only e PyTorch non
trovava una directory temporanea scrivibile. Non produsse generazione né fault
e non entra nell'esito Goal 12A.

## Conclusione

**CLEAN FAILURE.** Lo spegnimento controllato di rank 1 non lascia rank 0 in
hang. Rank 0 può terminare il calcolo locale già alimentato da dati ricevuti,
poi la successiva `send` Gloo rileva la chiusura del peer in circa 2,6 s dalla
terminazione effettiva di rank 1 (circa 12,9 s dall'avvio dello shutdown host),
senza raggiungere il timeout configurato. Anche il percorso di barrier/destroy
finale non aggiunge un blocco osservabile. Nessuna SIGILL o anomalia nativa è
correlata al risultato.
