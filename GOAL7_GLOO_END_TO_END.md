# Goal 7 — Gloo end-to-end reale

## Decisione

**GO — il modello reale funziona end-to-end sui due PC e si può passare al Goal 8 per tokenizer e generazione autoregressiva minima.**

Non sono stati implementati tokenizer, prompt, sampling, loop autoregressivo o Goal 8.

## Configurazione

- `torch.distributed`, backend `gloo`, CPU-only.
- Rank 0: atlas5, `192.168.1.128`; rank 1: argo3, `192.168.1.64`.
- `world_size=2`, rendezvous TCP su atlas5, porta configurabile 29500.
- Protocollo point-to-point bloccante `send`/`recv`; niente DDP, RPC o replica del modello.
- Un solo process group persistente per l'intera prova.
- Docker `--network host`, root filesystem read-only e checkpoint bind-mounted `:ro`.
- Nessun download è stato effettuato a runtime.

Prima di inizializzare Gloo ogni processo determina automaticamente l'IP sorgente della route verso il peer, lo associa all'interfaccia tramite `SIOCGIFADDR`, legge l'MTU da sysfs e imposta `GLOO_SOCKET_IFNAME`.

| nodo | interfaccia | IP locale | peer | MTU |
|---|---|---|---|---:|
| atlas5 | `eno1` | 192.168.1.128 | 192.168.1.64 | 1500 |
| argo3 | `eno1` | 192.168.1.64 | 192.168.1.128 | 1500 |

La route è quella Ethernet privata; le interfacce Wi-Fi 192.168.0.x non sono state selezionate. La porta 29500 è stata verificata libera prima dell'avvio e nuovamente libera dopo lo shutdown.

## Immagine Docker

La stessa immagine costruita su atlas5 è stata trasferita con `docker save | ssh ... docker load`, senza ricostruzione su argo3.

Image ID su entrambi i nodi:

`sha256:cc00a441fbb8f717ea6bea5eb084564aa1fd328b78da5a04e5b78a0add7bc9ad`

Test nell'immagine: **41 passed**. PyTorch: 2.13.0+cpu; quattro thread per stage; CUDA non utilizzata.

## Test Gloo senza modello

Il test iniziale ha completato:

1. rendezvous e inizializzazione;
2. invio atlas→argo di un piccolo tensore e conferma numerica;
3. round-trip di un hidden `[1,1,3584]` FP32 da 14,336 byte;
4. confronto bitwise esatto tramite view `int32`;
5. 5 warmup e 100 round-trip sullo stesso process group;
6. barrier e shutdown pulito.

| metrica hidden sintetico | valore |
|---|---:|
| round-trip mediano | 0.7357 ms |
| min / max | 0.4973 / 1.0091 ms |
| one-way stimato | 0.3679 ms |
| byte-identico | sì |

Il costo di rete è trascurabile rispetto ai circa 5.3 secondi di ciascuno stage.

## Caricamento e RAM

I moduli partono da `meta`; `RealTensorLoader` usa mmap tensor-by-tensor. Ogni processo usa un solo `QuantBuffer` e cache indipendenti. Il basso RSS subito dopo il load riflette il lazy paging mmap; i picchi reali sono misurati dopo i forward.

| nodo | tensori | byte logici | RSS/HWM prima | RSS/HWM dopo mapping | RSS finale | HWM massimo | MemAvailable finale | swap processo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| atlas5 | 295 | 15,384,983,040 | 211.73/211.73 MiB | 711.73/711.73 MiB | 13,528.14 MiB | 13,655.79 MiB | 28,198.23 MiB | 0 |
| argo3 | 296 | 15,384,997,376 | 211.24/211.24 MiB | 492.54/492.54 MiB | 15,521.59 MiB | 15,649.43 MiB | 29,726.05 MiB | 0 |

I picchi non superano quelli locali dei Goal 5/6. Non c'è crescita incontrollata: tra il secondo e il terzo forward RSS cresce di circa 19.5 MiB su atlas e 24.0 MiB su argo, ben sotto la guardia di 512 MiB. Nessun processo è entrato in swap.

## Confine atlas→argo

Il tensore reale prodotto da atlas ha:

- shape `[1,1,3584]`, FP32, 3,584 elementi, 14,336 byte;
- norma 213.813232421875;
- minimo -8.299798965454102, massimo 8.917988777160645;
- sentinelle `[-0.27306947, -0.78571272, -2.00668049, 2.10100007, 4.08258104]`.

Statistiche e sentinelle ricevute su argo sono identiche in tutte le prove. Nella prima prova argo ha restituito una sola eco diagnostica del tensore; atlas ha verificato uguaglianza bitwise esatta. Nelle prove 2 e 3 il percorso è stato normalmente unidirezionale e sono tornati soltanto gli 8 byte del token `int64`.

## Forward distribuito reale

Percorso validato:

`token 42 → embedding → layer 0–13 → Gloo → layer 14–27 → norm → lm_head → argmax → token int64 via Gloo`

Il token tecnico risultante è **320**. Non ha interpretazione linguistica.

### Tre prove indipendenti

Le cache vengono ricreate a posizione 4096 prima di ogni prova; tutti e 28 gli stati avanzano una sola volta a 4097. Non viene creato un nuovo process group.

| prova | AtlasStage | send hidden | Argo layer 14–27 | norm | lm_head | argmax | ritorno token | totale end-to-end | token |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5,359.003 ms | 0.143 ms | 5,267.710 ms | 0.239 ms | 114.387 ms | 0.631 ms | 0.145 ms | 10,744.667 ms | 320 |
| 2 | 5,342.174 ms | 0.110 ms | 5,229.817 ms | 0.194 ms | 109.035 ms | 0.388 ms | 0.163 ms | 10,683.879 ms | 320 |
| 3 | 5,269.728 ms | 0.131 ms | 5,162.644 ms | 0.230 ms | 109.251 ms | 0.485 ms | 0.143 ms | 10,544.603 ms | 320 |

Mediane:

- AtlasStage: 5,342.174 ms;
- chiamata `send` hidden: 0.131 ms (latenza effettiva separata misurata dal benchmark sintetico: ~0.368 ms one-way);
- ArgoStage layer: 5,229.817 ms;
- norm: 0.230 ms;
- lm_head: 109.251 ms;
- argmax: 0.485 ms;
- totale end-to-end: **10,683.879 ms**.

I tre hidden di confine sono identici, gli output sono finiti e il token risultante è sempre 320.

## Errori, timeout e shutdown

Le guardie rifiutano shape/dtype inattesi, NaN/Inf, superamento RSS, swap e crescita progressiva eccessiva. L'inizializzazione ha timeout configurabile; un test con rank remoto assente e timeout di 3 secondi ha terminato con `DistStoreError` dopo 4 secondi, exit code 1, senza listener o container residui.

Durante la prima esecuzione della variante con eco diagnostica argo3 è terminato una volta con SIGILL (exit 132) subito dopo l'eco. Rank 0 ha rilevato `Connection closed by peer`, è terminato senza restare appeso e non sono rimasti processi o porte occupate. La stessa immagine ha poi superato:

- test tensoriale PyTorch su argo3;
- test Gloo sintetico;
- ripetizione completa dei tre forward reali con eco bitwise.

L'anomalia non è risultata riproducibile e non ha richiesto cambio di librerie o codice numerico. Va comunque monitorata nel Goal 8; una nuova occorrenza renderebbe opportuno raccogliere core/kernel log e rivalutare l'ISA/runtime.

Entrambi i run finali riportano `clean_shutdown: true`; dopo le prove non risultano processi `run_atlas.py`/`run_argo.py` né listener sulle porte 29500/29501.

## File creati o modificati

- `src/spikingbrain_cpu/protocol.py`: route detection, init Gloo, validazione boundary e shutdown.
- `scripts/distributed_stage.py`: test sintetico, caricamento stage e protocollo reale.
- `scripts/run_atlas.py`, `scripts/run_argo.py`: entry point rank-specifici.
- `tests/test_protocol.py`: shape, dtype, finitezza e summary.
- `goal7_synthetic_atlas.json`, `goal7_synthetic_argo.json`: risultati Gloo sintetici.
- `goal7_real_atlas.json`, `goal7_real_argo.json`: risultati reali finali.
- `goal7_timeout.json`: test peer assente.
