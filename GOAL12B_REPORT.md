# Goal 12B — Physical Ethernet cable loss during active Gloo generation

## Esito

**TIMEOUT FAILURE (dual-rank, clean process exit).**

Lo scollegamento fisico di `eno1`, con entrambi gli host ancora accesi, non ha
prodotto una chiusura TCP immediatamente osservabile da Gloo. Rank 1 è rimasto
in `recv` e rank 0 in `send`; entrambi hanno raggiunto il timeout configurato di
75 secondi e sono terminati autonomamente con exit code 1. Non è rimasto alcun
processo, listener o container in esecuzione.

## Vincoli e integrità dell'esperimento

- Unico fault: Sergio ha scollegato manualmente il cavo Ethernet privato.
- Nessun firewall, `ip link down`, `kill`, `init 0`, spegnimento o fault
  injection software.
- Nessuna modifica a codice o repository; nessun commit o push; worktree
  rimasto pulito.
- Runtime esclusivo `spikingbrain-cpu:goal8.6-openblas`.
- Gloo vincolato a `eno1`; gestione remota di argo3 mantenuta via Wi-Fi.
- Rank e monitor non sono stati interrotti manualmente.
- Nessuna generazione è stata rilanciata dopo il ripristino del cavo.

## Preflight

### Rete

Prima del test entrambi i nodi riportavano:

- `/sys/class/net/eno1/carrier = 1`;
- `/sys/class/net/eno1/operstate = up`;
- flag `UP,LOWER_UP`, MTU 1500;
- route privata diretta su `eno1`.

| Funzione | Atlas5 | Argo3 |
|---|---|---|
| Gloo Ethernet | `eno1`, 192.168.1.128 | `eno1`, 192.168.1.64 |
| Gestione Wi-Fi | 192.168.0.89 | 192.168.0.247 |

La prova SSH registra esplicitamente:

```text
SSH_CONNECTION=192.168.0.89 ... 192.168.0.247 22
```

Sullo stesso host remoto, `ip route get 192.168.1.128` indicava invece
`dev eno1 src 192.168.1.64`. Quindi gestione e traffico Gloo usavano percorsi
distinti. Dopo il link-down la sessione Wi-Fi ha continuato a funzionare.

### Runtime, sorgenti e risorse

- Immagine su entrambi:
  `sha256:696d98683cee6eef018a73e5c70b402f9103fc24b230356975a7ef1685aed836`.
- PyTorch stabile OpenBLAS Ivy Bridge, niente MKL/MKLDNN/CUDA.
- Hash `distributed_generate.py` sui due nodi:
  `b37eb401ab870ae2de6747934c099598d78abf27e9e9aa3fd295140377dcf9a3`.
- Hash `protocol.py` sui due nodi:
  `2cfdb5730b02a617efd9eecbc3254e5252f7de3dacc064f5a3cfd93fa92b8962`.
- Checkpoint e tokenizer presenti e montati read-only.
- Circa 28–30 GiB di memoria disponibile, swap 0 e ampio spazio disco.
- Porta 29622 libera; nessun processo o container Goal 12 in esecuzione.
  Erano conservati soltanto i container fermi di Goal 12A.

## Configurazione esatta

| Voce | Valore |
|---|---|
| Data | 2026-08-16, Europe/Rome (CEST, UTC+02:00) |
| Backend | Gloo CPU, world size 2 |
| Rendezvous | 192.168.1.128:29622 |
| Interfaccia Gloo | `eno1` |
| Docker networking | host |
| Partizione | atlas5: embedding/layer 0–13; argo3: layer 14–27/norm/`lm_head` |
| Thread | 4 per rank |
| Timeout | 75 s |
| `max_prompt_tokens` | 64 |
| `max_new_tokens` | 40 |
| `max_bytes` | 16,000,000,000 per rank |
| Prompt tokenizzati | 31 |
| Decoding | greedy argmax |

Prompt:

> Explain in detail how two old computers can cooperate to run a large language model, describing the flow of data between the machines and the main technical limitations.

## Comandi dei rank

Rank 1 su argo3, avviato via percorso di gestione Wi-Fi ma con Gloo su `eno1`:

```bash
docker run --name goal12b-argo-rank1 --network host \
  -e GLOO_SOCKET_IFNAME=eno1 \
  -v /home/sergio/Modelli/SpikingBrain/SpikingBrain-7B-W8ASpike:/model:ro \
  -v /home/sergio/goal12a-20260816/runtime/src:/app/src:ro \
  -v /home/sergio/goal12a-20260816/runtime/scripts:/app/scripts:ro \
  spikingbrain-cpu:goal8.6-openblas \
  python scripts/generate_argo.py \
  --master-addr 192.168.1.128 --port 29622 --peer 192.168.1.128 \
  --model-dir /model --max-bytes 16000000000 \
  --max-prompt-tokens 64 --max-new-tokens 40 --threads 4 --timeout 75
```

Rank 0 su atlas5:

```bash
docker run --name goal12b-atlas-rank0 --network host \
  -e GLOO_SOCKET_IFNAME=eno1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v /home/sergio/Modelli/SpikingBrain/SpikingBrain-7B-W8ASpike-atlas:/model:ro \
  -v /home/sergio/Modelli/SpikingBrain/SpikingBrain-7B-W8ASpike-tokenizer:/tokenizer:ro \
  -v /home/sergio/Progetti/spikingbrain-cluster/src:/app/src:ro \
  -v /home/sergio/Progetti/spikingbrain-cluster/scripts:/app/scripts:ro \
  spikingbrain-cpu:goal8.6-openblas \
  python scripts/generate_atlas.py \
  --master-addr 192.168.1.128 --port 29622 --peer 192.168.1.64 \
  --model-dir /model --tokenizer-dir /tokenizer \
  --prompt 'Explain in detail how two old computers can cooperate to run a large language model, describing the flow of data between the machines and the main technical limitations.' \
  --max-bytes 16000000000 --max-prompt-tokens 64 \
  --max-new-tokens 40 --threads 4 --timeout 75
```

## Monitor e qualità temporale

Prima dei rank sono stati avviati:

- monitor carrier/operstate/link ad alta frequenza su entrambi;
- monitor ping Ethernet, container, PID, CPU e RSS;
- osservatore congiunto interrogato via Wi-Fi;
- journal kernel/NetworkManager come fonte indipendente.

Il monitor carrier a 10 Hz di argo3 è completo. Quello dedicato di atlas5 si è
chiuso prematuramente prima del workload per un limite della sessione di
orchestrazione locale; questa anomalia è conservata nei log e non viene
nascosta. Atlas5 resta coperto dall'osservatore congiunto e soprattutto dal
journal kernel a precisione microsecondo. Il fault time non deriva mai
dall'orario dell'avviso a Sergio.

Bracket a 10 Hz su argo3:

```text
15:16:26.319336804  carrier=1 operstate=up
15:16:26.452333899  carrier=0 operstate=down
```

Precisione del bracket: 132,997 ms. Timestamp operativo adottato per le latenze:
**15:16:26.452333899 CEST**, primo campione certo a carrier 0.

Conferme kernel:

- argo3: `NIC Link is Down` alle **15:16:26.333509**;
- atlas5: `NIC Link is Down` alle **15:16:26.730568**.

La piccola differenza riflette rilevamento/clock delle due NIC e non cambia la
classificazione.

## Prova di generazione prima del fault

Il segnale a Sergio è stato dato soltanto dopo la seguente sequenza osservata:

1. entrambi i process group inizializzati;
2. stage atlas materializzato (~13,5 GiB RSS durante il forward);
3. stage argo materializzato (~15,6 GiB RSS durante il forward);
4. prefill Atlas inviato e forward reale Argo completato;
5. token 1 ricevuto da Atlas;
6. decode Atlas successivo completato e nuovo hidden consegnato ad Argo.

Al carrier-down erano stati completati **2 token**. Rank 0 stava elaborando
localmente il token 2 per produrre l'hidden successivo. Gli ID testuali non sono
recuperabili: il driver inserisce gli step soltanto nel JSON di successo finale,
che non viene prodotto nel percorso di eccezione. Il conteggio è ricavato dalla
sequenza alternata CPU/Gloo e dal punto dei traceback.

## Cronologia del fault

| Timestamp CEST | Evento |
|---|---|
| 15:14:33.358 | Container rank 1 avviato |
| 15:14:35.213 | Container rank 0 avviato |
| 15:16:05–15:16:13 | Token 1 ricevuto/elaborato; decode autoregressivo seguente avviato su Argo |
| ~15:16:22.987 | Rank 1 entra nel `recv` dell'header hidden successivo |
| 15:16:26.319 | Ultimo campione carrier 1 su argo3 |
| 15:16:26.452 | Primo campione carrier 0/down su argo3: fault time adottato |
| 15:16:26.731 | Kernel atlas5: NIC link down |
| ~15:16:29.682 | Rank 0 entra nel `send` del payload hidden |
| 15:17:37.987 | Rank 1 emette il primo errore Gloo |
| 15:17:38.600 | Container rank 1 terminato, exit 1 |
| 15:17:44.682 | Rank 0 emette il proprio errore Gloo |
| 15:17:45.438 | Container rank 0 terminato, exit 1 |
| 15:19:48.335 | Link argo3 nuovamente up, 1 Gbps full duplex |
| 15:19:48.977 | Link atlas5 nuovamente up, 1 Gbps full duplex |

I tempi stimati di ingresso nelle operazioni derivano dal timestamp dell'errore
meno i 75,000 s dichiarati esplicitamente da Gloo.

## Rank che rileva per primo e operazioni Gloo

### Rank 1 / argo3 — primo failure

Operazione:

```text
argo() -> recv_hidden() -> dist.recv(header, peer) -> work.wait()
```

Errore esatto:

```text
RuntimeError: [/src/third_party/gloo/gloo/transport/tcp/unbound_buffer.cc:78]
Timed out waiting 75000ms for recv operation to complete
```

- Errore emesso: 15:17:37.986695477.
- Latenza da primo carrier 0: **71,534361578 s**.
- L'operazione era già in attesa da circa 3,466 s quando il cavo è stato
  scollegato; per questo la latenza dal carrier-down è minore di 75 s.
- Exit code: **1**, `OOMKilled=false`.

### Rank 0 / atlas5 — secondo failure

`send_hidden()` invia prima un header e poi il payload. Il traceback è sul
secondo invio:

```text
atlas() -> send_hidden(next_hidden, 1)
        -> dist.send(hidden.contiguous(), peer) -> work.wait()
```

Errore esatto:

```text
RuntimeError: [/src/third_party/gloo/gloo/transport/tcp/unbound_buffer.cc:129]
Timed out waiting 75000ms for send operation to complete
```

- Errore emesso: 15:17:44.681905653.
- Latenza da primo carrier 0: **78,229571754 s**.
- Il `send` bloccante è iniziato circa 3,230 s dopo il carrier-down, dopo che
  rank 0 aveva terminato il proprio calcolo locale.
- Exit code: **1**, `OOMKilled=false`.

Rank 1 ha quindi segnalato il problema **6,695210176 s** prima di rank 0, non
per un meccanismo di rilevamento diverso, ma perché il suo `recv` era iniziato
prima del `send` di rank 0. Entrambi hanno atteso esattamente il timeout della
propria operazione.

## Timeout e comportamento TCP/Gloo

La perdita di carrier non ha generato `Connection closed by peer`: entrambi gli
host e processi erano ancora vivi e nessun endpoint ha chiuso ordinatamente il
socket. Le operazioni sono rimaste in attesa fino a:

- `Timed out waiting 75000ms for recv operation to complete` su rank 1;
- `Timed out waiting 75000ms for send operation to complete` su rank 0.

Il timeout di 75 s è dunque stato raggiunto su entrambi i rank. Non si tratta di
immediate link failure e non è un hang indefinito.

## `shutdown()` / barrier / destroy process group

Su entrambi i rank il driver:

1. cattura l'eccezione applicativa;
2. stampa il JSON di errore;
3. entra nel `finally` e chiama `shutdown()`;
4. tenta `barrier()` e garantisce `destroy_process_group()`;
5. rilancia l'eccezione originale.

Tempi osservati:

- rank 1: JSON alle 15:17:37.9867, traceback del re-raise alle 15:17:37.9929
  (~6 ms), container terminato alle 15:17:38.5999;
- rank 0: JSON alle 15:17:44.6819, traceback alle 15:17:44.6836 (~2 ms),
  container terminato alle 15:17:45.4381.

La barrier finale non ha aggiunto un secondo timeout né un hang; il process
group già in errore ha fatto fallire/rientrare rapidamente la barrier e
`destroy_process_group()` è stato eseguito. Come in Goal 12A, il testo esatto di
`shutdown_error` non appare perché il report viene stampato prima che il
`finally` aggiunga quel campo e non viene ristampato nel percorso di re-raise.

## Cleanup e anomalie native

Prima della riconnessione e dopo il ripristino:

- nessun processo `distributed_generate`, `generate_atlas` o `generate_argo`;
- nessun listener sulla porta 29622;
- nessun container Goal 12B in esecuzione;
- container conservati fermi per ispezione: entrambi `Exited (1)`;
- swap 0 su entrambi;
- `SIGILL`: 0;
- `invalid opcode`: 0;
- segfault/trap/`libtorch_cpu.so`: 0;
- OOM/`Killed process`: 0.

Dopo il ripristino entrambi riportavano `carrier=1`, `operstate=up`, route
192.168.1.x diretta su `eno1` e ping Ethernet 3/3 (~0,3 ms). La sessione di
controllo argo3 continuava a dichiarare `SSH_CONNECTION` sulla rete Wi-Fi.

## Confronto Goal 12A / Goal 12B

| Aspetto | Goal 12A: shutdown host | Goal 12B: cavo fisico |
|---|---|---|
| Host peer | Spegnimento ordinato, processo infine terminato | Entrambi sempre accesi |
| Stato TCP | Peer/processo chiude o scompare | Socket vivo ma percorso senza carrier |
| Primo errore | Rank 0, `Connection closed by peer` | Rank 1, timeout `recv` |
| Altro rank | Rank 1 terminato dallo shutdown (137) | Rank 0 timeout `send` |
| Timeout Gloo | Non raggiunto | Raggiunto su entrambi, 75,000 s |
| Latenza rilevante | ~2,612 s da morte effettiva rank 1 a errore rank 0 | 71,534 s carrier→rank 1; 78,230 s carrier→rank 0 |
| Classificazione | CLEAN FAILURE | TIMEOUT FAILURE |
| Residui/crash nativi | Nessuno | Nessuno |

La differenza principale è causale: Goal 12A produce una chiusura del peer
visibile allo stack TCP; Goal 12B produce una perdita silenziosa del mezzo
fisico, quindi Gloo può soltanto attendere la scadenza delle operazioni.

## Artefatti

Directory: `/home/sergio/goal12b-20260816/`

- preflight atlas5 e argo3 via Wi-Fi;
- prova delle route separate e hash sorgenti;
- confronto orologi;
- `valid/config.txt`;
- stdout, stderr, timestamp ed exit code separati per rank;
- log Docker timestampati e inspect completi;
- `carrier_argo.tsv` a 10 Hz;
- observer congiunto e monitor di stato/ping/CPU/RSS;
- log kernel e NetworkManager del link-down/link-up;
- controlli cleanup pre/post riconnessione.

## Conclusione

**TIMEOUT FAILURE.** La perdita fisica di Ethernet durante generazione attiva
non viene rilevata immediatamente da Gloo in questa configurazione. Rank 1,
bloccato sul `recv` dell'header successivo, e rank 0, bloccato sul `send` del
payload hidden, scadono ciascuno dopo 75 secondi dalla propria operazione. Rank
1 termina per primo perché la sua attesa era iniziata prima del link-down. Gli
exit sono puliti a livello di processo (code 1), la barrier finale non introduce
un secondo blocco, non restano risorse attive e non emergono crash nativi.
