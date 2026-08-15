# Goal 6 — AtlasStage reale

## Esito

**GO — AtlasStage e ArgoStage sono entrambi validati localmente; si può passare al Goal 7 per collegarli tramite Gloo.**

Il Goal 6 non ha implementato Gloo, non ha eseguito forward distribuiti e non ha usato tokenizer o generazione.

## Ambiente e spazio disco

È stata costruita su atlas5 l'immagine CPU-only `spikingbrain-cpu:goal6` e trasferita ad argo3 con `docker save | ssh ... docker load`. Entrambi i nodi hanno eseguito lo stesso image ID:

`sha256:13088d30276eb6969ddf91e499c82241578148703d3485bff0b4e5163072542f`

Prima del trasferimento:

| nodo | filesystem | dimensione | usato | disponibile | checkpoint/progetto |
|---|---:|---:|---:|---:|---:|
| atlas5 | `/dev/sda2` | 439 GiB | 144 GiB | 273 GiB | progetto 396 KiB; immagine Goal 5 290,576,761 B |
| argo3 | `/dev/sda2` | 439 GiB | 202 GiB | 215 GiB | checkpoint originale 29 GiB |

Docker occupava 7.294 GB di immagini su atlas5 e 14.41 GB su argo3. Il margine era ampio sia per lo staging sia per la destinazione; non è stato necessario usare swap come memoria del modello.

## Piano atlas-only

Il planner ha validato, prima di leggere i dati tensoriali:

- `model.embeddings.weight`;
- tutti e soli i tensori `model.layers.0.*` … `model.layers.13.*`;
- 295 tensori FP32;
- 15,384,983,040 byte logici, cioè 14.328382 GiB.

Sono stati esclusi esplicitamente layer 14–27, `model.norm.weight` e `lm_head.weight`. Il manifest machine-readable `atlas_tensor_manifest.json` registra nome, shard sorgente, shape, dtype e byte di ciascun tensore.

## Creazione e trasferimento del subset

`scripts/create_atlas_subset.py` legge gli header Safetensors, ricostruisce quattro shard atlas-only e copia gli intervalli tensoriali con buffer da 8 MiB. Lavora uno shard sorgente alla volta, non usa `safe_open().get_tensor()`, non crea un `state_dict` e non mantiene i 14 GiB in RAM. I file temporanei `.partial` sono creati fuori dal checkpoint originale e rinominati solo dopo flush/fsync.

Lo staging è stato creato su argo3 in `$HOME/SpikingBrain-goal6-staging`. Il checkpoint atlas è stato trasferito sulla rete privata, da `192.168.1.64`, con `rsync`; 15,385,111,631 byte sono stati trasferiti in circa 2m22s, circa 103 MB/s. Destinazione:

`$HOME/Modelli/SpikingBrain/SpikingBrain-7B-W8ASpike-atlas`

Dimensione complessiva dei file coperti dal manifest: 15,385,110,355 byte (14.328501 GiB), oltre al piccolo `checksums.json`.

| file | byte | SHA-256 |
|---|---:|---|
| `model-00001-of-00004.safetensors` | 4,739,635,424 | `4aa1b818ec1fa63f3aa3c1a5f1a88306f39e67ba6d054f709913097f43cd679d` |
| `model-00002-of-00004.safetensors` | 4,838,173,600 | `452d45639eb70156f67bd7f78a57d3985a43249a1f9479f743cd8fcbf9bc88d6` |
| `model-00003-of-00004.safetensors` | 821,109,496 | `19ec6a63519d5bfaa098b0d6881ac4dea3a4b5cce8e76d52cfa074263be817c5` |
| `model-00004-of-00004.safetensors` | 4,986,098,640 | `d4e3fe2aa5ab29a79659562b4f0ef3b0d6392085df75ac54656d84f3f698ac69` |
| indice | 24,615 | `43bc0c5a18e5f2a8eedf99693c09be328f00cf821c14970c0e001d1ae720fe6d` |
| config | 1,005 | `6f0843853d2686808dea19fe715d4958dcff0c7007da587821b1890ed4bd8466` |
| manifest tensori | 67,575 | `5dbfca3587bfb94aa036dada3b9e2d5bc9ab12e6717defbaed4f1a98cca59811` |

Tutti gli SHA-256 sono stati calcolati prima del trasferimento e ricalcolati su atlas5. Anche `checksums.json` coincide tra sorgente e destinazione (`b1463627…d1ab7c6`).

## AtlasStage e caricamento incrementale

`AtlasStage` in `src/spikingbrain_cpu/model_partition.py` contiene embedding e layer globali 0–13, inizialmente su `meta`. `RealTensorLoader` materializza direttamente i singoli tensori dal subset mmap. Il container è stato eseguito con root filesystem read-only, rete disabilitata e checkpoint bind-mounted `:ro`.

| stato residente | tensori | byte logici | RSS MiB | HWM MiB | MemFree MiB | MemAvailable MiB | swap processo |
|---|---:|---:|---:|---:|---:|---:|---:|
| embedding | 1 | 2,179,989,504 | 2,362.78 | 2,362.78 | 8,211.36 | 29,063.66 | 0 |
| embedding + 1 layer | 24 | 3,126,903,296 | 3,265.86 | 3,265.86 | 7,302.82 | 29,058.12 | 0 |
| embedding + 2 layer | 43 | 4,066,417,152 | 4,163.86 | 4,163.86 | 6,865.77 | 29,060.04 | 0 |
| embedding + 4 layer | 85 | 5,952,844,800 | 5,965.20 | 5,965.20 | 6,863.78 | 29,059.05 | 0 |
| embedding + 8 layer | 169 | 9,725,700,096 | 9,564.10 | 9,564.10 | 6,858.00 | 29,053.27 | 0 |
| embedding + 14 layer | 295 | 15,384,983,040 | 14,967.44 | 14,967.44 | 6,848.55 | 29,052.42 | 0 |

Il conteggio finale del loader coincide esattamente col piano: 295 tensori e 15,384,983,040 byte. Non è emersa crescita RSS anomala rispetto ai pesi residenti.

## QuantBuffer e cache

- Un solo `QuantBuffer` condiviso tra tutti i layer (`identity_count = 1`).
- Dimensione buffer: 271,581,184 byte (259.0 MiB).
- RSS: 14,967.44 → 15,226.53 MiB dopo l'allocazione.
- 14 cache indipendenti: sette `GLACache` per i layer pari e sette `KVCache` per i dispari.
- Cache totali a posizione 4096: 130,285,568 byte (124.25 MiB).
- Gli indici globali sono 0–13; tutte le posizioni sono avanzate coerentemente da 4096 a 4097.

## Forward e benchmark

Configurazione: PyTorch 2.13.0+cpu, quattro thread, `torch.inference_mode()`, un warmup e tre iterazioni. Token tecnico: 42.

| misura | risultato |
|---|---:|
| embedding mediana | 0.020 ms |
| primo forward profilato layer 0–13 | 5,425.036 ms |
| media layer GLA | 357.750 ms |
| media layer sliding-window attention | 416.810 ms |
| benchmark stage mediana | 5,465.185 ms |
| benchmark min / max | 5,404.869 / 5,525.840 ms |
| token → embedding → layer 0–13, singola esecuzione | 5,683.570 ms |

La mediana è circa il 2.5% sopra la previsione Goal 3 di 5.33 s, una corrispondenza sufficientemente stretta. L'output finale ha shape `[1,1,3584]`, dtype FP32 ed è interamente finito; tutti gli output intermedi sono finiti.

## RAM finale e integrità

- RSS finale: 15,739.59 MiB.
- HWM massimo: 15,867.30 MiB.
- MemFree host finale: 5,958.86 MiB.
- MemAvailable host finale: 28,163.13 MiB.
- Swap del processo: 0 MiB.

L'ampio scarto tra MemFree e MemAvailable deriva principalmente dalla page cache mmap recuperabile. Il margine prudenziale richiesto di 4 GiB è rispettato.

Il checkpoint originale argo3 conserva gli stessi nomi, dimensioni e mtime registrati prima dell'estrazione. Gli hash post-operazione di config e indice sono rispettivamente `6f084385…d8466` e `9e73b824…8f88`, identici ai valori iniziali. Il subset è stato verificato integralmente prima e dopo il trasferimento e montato read-only durante il forward. Non sono stati creati full state dict, copie dense permanenti o un secondo QuantBuffer; non sono stati usati CUDA, Triton, fla o FlashAttention e non è stata effettuata quantizzazione permanente.

## File aggiunti o modificati

- `src/spikingbrain_cpu/model_partition.py`: `AtlasStage`.
- `scripts/create_atlas_subset.py`: planner ed estrattore streaming atlas-only.
- `scripts/verify_subset.py`: verifica SHA-256 streaming.
- `scripts/build_atlas_stage.py`: caricamento incrementale, controlli RAM/cache e benchmark.
- `tests/test_model_partition.py`: test struttura meta, buffer, cache e guardie AtlasStage.
- `goal6_atlas_results.json`: risultati machine-readable.

Test Docker complessivi: **38 passed**. Il solo warning è l'assenza intenzionale di NumPy, non usato dal percorso di inferenza.
