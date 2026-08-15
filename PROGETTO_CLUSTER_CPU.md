# Progetto del cluster CPU per SpikingBrain-7B-W8ASpike

> **Historical initial plan.** This document records the original 16/12
> partition proposal. Full-block benchmarks in Goal 3 subsequently replaced
> it with the final 14/14 split used by Goals 4–9. It remains available as part
> of the engineering history.

## Esito sintetico

La partizione del modello su due nodi è tecnicamente possibile e l'indice Safetensors consente di assegnare i tensori per layer senza modificare gli originali. La configurazione consigliata è:

- **atlas5 (coordinatore/rank 0):** embedding e layer `0..15`;
- **argo3 (worker/rank 1):** layer `16..27`, norm finale e `lm_head`;
- **comunicazione:** processo persistente per nodo, `torch.distributed` con backend Gloo e protocollo applicativo minimale basato su `send`/`recv` (o `isend`/`irecv`), non DDP;
- **un solo confine di pipeline:** l'hidden state passa da atlas5 ad argo3; il token scelto/logit necessario torna al coordinatore.

Questa scelta assegna più calcolo al più veloce i5-4590/AVX2, ma conserva margine RAM su entrambi i nodi. Prima di qualsiasi inferenza è indispensabile realizzare e validare i fallback CPU dei kernel del modello.

## Analisi dell'indice e dell'architettura

Analisi effettuata in sola lettura su argo3. Non sono stati caricati dati tensoriali: per le dimensioni esatte sono stati letti soltanto l'indice JSON e gli header Safetensors (`dtype`, shape e `data_offsets`).

`model.safetensors.index.json` contiene:

- `metadata.total_size = 30.769.980.416` byte = **28,6568 GiB**;
- **591 tensori**, tutti mappati su 7 shard;
- embedding: `model.embeddings.weight`;
- blocchi: `model.layers.0.*` fino a `model.layers.27.*`;
- uscita: `model.norm.weight` e `lm_head.weight`.

Da `config.json`: hidden size 3584, 28 layer, intermediate size 18944, 28 teste, vocabolario 152064. I layer dispari `1,3,...,27` sono sliding-window/FlashAttention; quelli pari sono Gated Linear Attention (GLA). I blocchi sono quindi già numerati e sequenziali e si possono dividere a qualunque confine tra due layer.

Dimensioni esatte dei gruppi:

| Gruppo | Quantità | Byte per elemento | GiB per elemento | GiB totali |
|---|---:|---:|---:|---:|
| embedding | 1 | 2.179.989.504 | 2,030273 | 2,030273 |
| layer GLA pari | 14 | 946.913.792 | 0,881882 | 12,346348 |
| layer attention dispari | 14 | 939.513.856 | 0,874990 | 12,249860 |
| norm finale | 1 | 14.336 | 0,000013 | 0,000013 |
| `lm_head` | 1 | 2.179.989.504 | 2,030273 | 2,030273 |

Il totale derivato dagli header coincide esattamente con `metadata.total_size`. La dicitura W8ASpike non significa che i pesi su disco siano int8: in questo checkpoint sono FP32; i buffer `weight_quantizer.scales` sono inclusi nei conteggi sopra.

Gli shard non rispettano sempre i confini dei layer: i layer 2, 8, 13, 18 e 23 attraversano due shard. Questo non impedisce la partizione logica, ma il loader del Goal 2 deve selezionare i tensori per prefisso, non assumere “uno shard = un gruppo di layer”. Con il taglio consigliato atlas5 avrà bisogno in futuro dei dati presenti negli shard 1–4; lo shard 4 contiene anche tensori non destinati ad atlas5, che non dovranno essere materializzati in RAM.

## Partizione raccomandata e RAM

| Nodo | Moduli | Pesi FP32 | RAM disponibile indicata | Margine teorico dopo i soli pesi |
|---|---|---:|---:|---:|
| atlas5 | embedding + layer `0..15` | **16,0853 GiB** | ~29 GiB | **~12,91 GiB** |
| argo3 | layer `16..27` + norm + `lm_head` | **12,5715 GiB** | ~30 GiB | **~17,43 GiB** |

Il margine è una stima superiore: vanno ancora sottratti sistema operativo, runtime Python/PyTorch, allocator, hidden states e cache GLA/KV. Durante il caricamento bisogna evitare copie temporanee dell'intero `state_dict`; costruzione su device `meta`, caricamento tensor-by-tensor e `inference_mode()` sono requisiti, non ottimizzazioni facoltative.

Il taglio perfettamente bilanciato dopo il layer 13 produrrebbe 14,3284 GiB per nodo (margini ~14,67 e ~15,67 GiB), ma lascerebbe metà dei blocchi al processore più lento. Il taglio 16/12 sacrifica circa 1,4 GiB di margine su atlas5 per una distribuzione del calcolo più sensata. Il punto esatto andrà confermato con benchmark di singolo layer nel Goal 2; 14/14 resta il fallback se il picco RAM di atlas5 risultasse eccessivo.

## Traffico di rete

Al confine il tensore ha shape `[batch, sequenza, 3584]`. In FP32:

`byte = batch × token_passati × 3584 × 4 = batch × token_passati × 14.336`.

| Caso, batch 1 | Dati atlas5 → argo3 | Solo serializzazione a 934 Mbit/s |
|---|---:|---:|
| decoding, 1 nuovo token | **14 KiB** | ~0,12 ms |
| prefill, 1.024 token | **14 MiB** | ~0,13 s |
| prefill, 4.096 token | **56 MiB** | ~0,50 s |
| prefill, 131.072 token teorici | **1,75 GiB** | ~16,1 s |

A questi tempi si aggiungono circa 0,3 ms di latenza e overhead di protocollo/sincronizzazione. In BF16 i valori si dimezzerebbero, ma non va assunto finché i fallback CPU e la precisione non sono validati. Dopo l'elaborazione su argo3, per la generazione conviene trasferire ad atlas5 soltanto il token selezionato e le informazioni minime di controllo; rimandare tutti i logits del vocabolario costerebbe circa 594 KiB FP32 per token. In alternativa sampling e criteri di arresto possono risiedere su argo3, mentre atlas5 resta coordinatore della sessione.

Ogni nodo deve mantenere localmente soltanto la cache dei propri layer. Cache, attention mask, `position_ids` e `cache_position` richiedono un protocollo coerente, ma le cache non devono attraversare la rete a ogni token.

## Tecnologia

Scelta: **pipeline custom a due stadi sopra `torch.distributed`/Gloo**.

- Gloo è adatto ai tensori CPU e consente un rendezvous TCP semplice sulla rete privata.
- Non serve DDP: DDP replica il modello e sincronizza gradienti, il contrario dell'obiettivo.
- La pipeline generica di PyTorch aggiunge integrazione e scheduling non necessari per due stadi, batch 1 e inferenza autoregressiva, oltre a non conoscere la cache custom di questo modello.
- RPC/TensorPipe renderebbe comode le chiamate remote, ma aggiunge serializzazione, lifecycle e gestione degli errori. Per un solo trasferimento tensoriale con protocollo statico, `send`/`recv` è più piccolo, osservabile e facile da diagnosticare.
- MPI non offre un vantaggio sufficiente rispetto a Gloo e introdurrebbe un'ulteriore dipendenza di sistema.

atlas5 deve essere **coordinatore/rank 0**: ospita tokenizer, embedding, prompt/session loop e i primi layer; apre il rendezvous su `192.168.1.128`. argo3 è rank 1 persistente e conserva cache, layer finali, norm, head e sampling. Non va avviato un processo SSH nuovo per token.

## Bloccanti CPU nel codice attuale

La distribuzione non risolve l'incompatibilità CPU. Il checkpoint contiene codice remoto che oggi dipende direttamente da percorsi GPU:

1. `modeling_gla_swa.py` importa `fla` (`RMSNorm`, `ShortConvolution`, `swish`) e `swiglu_linear` da `activations.py`.
2. `activations.py` importa Triton senza guardia e crea funzioni tramite `torch.cuda.jiterator`; lo SwiGLU usato dai blocchi passa da questo file.
3. `gla_attention.py` importa e chiama `chunk_gla`, `fused_chunk_gla` e, per il decoding a un token, `fused_recurrent_gla`. Questi operatori `fla` sono il percorso normale, non opzionale.
4. `window_attention.py` chiama `flash_attn_func`/`flash_attn_varlen_func`; se FlashAttention 2 non è disponibile, i simboli non vengono importati ma il `forward` li usa comunque.
5. QuantLinear arrotonda/dequantizza i pesi a runtime e poi usa `torch.nn.functional.linear`; i pesi restano FP32 e questo può essere estremamente lento e creare temporanei. W8ASpike, così implementato, non dà automaticamente storage o GEMM int8 CPU.

Prima del test distribuito servono implementazioni CPU pure PyTorch, numericamente confrontabili: RMSNorm/SwiGLU; sliding-window scaled-dot-product attention senza FlashAttention; GLA recurrent/chunk senza Triton; e una strategia per evitare copie FP32 transitorie nella quantizzazione. È preferibile mantenere questi adattamenti in una copia/versione del codice nel progetto, lasciando intatti i file del modello.

## Deliverable del Goal 2

File/script suggeriti:

- `src/spikingbrain_cpu/ops.py`: fallback CPU per RMSNorm, SwiGLU, GLA e sliding-window attention;
- `src/spikingbrain_cpu/model_partition.py`: classi `AtlasStage` e `ArgoStage`, con layer range configurabile;
- `src/spikingbrain_cpu/selective_loader.py`: lettura per prefisso dall'indice, materializzazione diretta e controllo byte/dtype; mai full `state_dict`;
- `src/spikingbrain_cpu/protocol.py`: header dei messaggi, shape/dtype, step, mask/position e gestione EOS/errori;
- `run_atlas.py`: rank 0, tokenizer, sessione, primo stadio e coordinamento;
- `run_argo.py`: rank 1, secondo stadio, cache, norm/head e sampling;
- `configs/cluster.toml`: IP, porta, timeout, layer `0:16` e `16:28`, thread CPU;
- `scripts/inspect_checkpoint.py`: ripetizione verificabile dell'analisi header senza caricare pesi;
- `tests/test_cpu_ops.py`, `tests/test_partition_equivalence.py`, `tests/test_protocol_loopback.py`: test su tensori piccoli/sintetici prima dei pesi reali;
- `scripts/stage_atlas_files.sh`: solo dopo approvazione, copia controllata dei file necessari su atlas5 con checksum, senza alterare gli originali.

Ordine consigliato: fallback CPU e test sintetici; benchmark di un layer GLA e uno attention su entrambi i nodi; loader selettivo; test loopback locale a due processi; test rete con tensori sintetici; solo infine staging dei pesi e smoke test.

## Raccomandazione finale

**Procedere con il cluster a due nodi, ma solo come prototipo sperimentale CPU e con un gate obbligatorio sui fallback CPU.** La partizione risolve realisticamente il vincolo di capacità RAM e la Gigabit Ethernet è adeguata a un unico confine. Non garantisce però prestazioni interattive: i 28,66 GiB FP32, il processore Ivy Bridge di argo3 e soprattutto l'assenza attuale di kernel CPU sono colli di bottiglia più importanti della rete. Se i benchmark del Goal 2 mostrano che i fallback GLA/attention o QuantLinear sono impraticabili, la decisione corretta sarà fermare l'inferenza FP32 distribuita e valutare una conversione/quantizzazione CPU realmente supportata.
