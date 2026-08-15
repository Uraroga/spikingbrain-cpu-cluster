# Goal 8R — Ripetizione stabile della generazione testuale

## Decisione finale

**GO-GENERATION-STABLE**: 5/5 sessioni complete e indipendenti, stessi token greedy, 0 SIGILL, 0 invalid opcode, 0 trap in `libtorch_cpu.so`, 0 swap di processo, cache coerenti su tutti i 28 layer e shutdown pulito dopo ogni sessione.

Non è stato avviato Goal 9. Non sono stati ricompilati PyTorch o OpenBLAS, non è stato modificato il runtime, non sono stati introdotti MKL/MKLDNN e la partizione è rimasta 14/14.

## Immagine e preflight

L'immagine usata esclusivamente su entrambi i nodi è `spikingbrain-cpu:goal8.6-openblas` (368.007.299 byte).

| nodo | Image ID | esito |
|---|---|---|
| atlas5 | `sha256:696d98683cee6eef018a73e5c70b402f9103fc24b230356975a7ef1685aed836` | identico all'atteso |
| argo3 | `sha256:696d98683cee6eef018a73e5c70b402f9103fc24b230356975a7ef1685aed836` | identico all'atteso |

Il preflight ha riportato su entrambi `torch.__version__ = 2.13.0+openblas.ivybridge`, `torch.backends.mkl.is_available() = False`, `torch.backends.mkldnn.is_available() = False` e `torch.cuda.is_available() = False`. La capability CPU è `AVX2` su atlas5 e `DEFAULT` su argo3, come atteso. I microtest `exp`, `logsigmoid` e matmul FP32 hanno prodotto risultati finiti.

`torch.__config__.show()` dichiara `BLAS_INFO=open`, `LAPACK_INFO=open`, `USE_MKL=0`, `USE_MKLDNN=0` e `USE_GLOO=1`. Su entrambi `ldd libtorch_cpu.so` risolve `libopenblas.so.0`; non compare alcuna dipendenza MKL/MKLDNN.

Configurazione invariata: `torch.distributed`, Gloo CPU-only, `--network host`, `GLOO_SOCKET_IFNAME=eno1`, quattro thread per rank, checkpoint/tokenizer montati read-only, atlas5 rank 0 con embedding e layer 0–13, argo3 rank 1 con layer 14–27, norm e `lm_head`.

## Prompt e tokenizer

Il prompt esatto è `Hello`, senza inserimento manuale di BOS o token speciali:

- token prompt: `[9707]`, lunghezza 1;
- round-trip: `Hello`;
- BOS effettivo: `<|im_start|>` / 151644, ma `add_bos_token=false`;
- EOS effettivo: `<|endoftext|>` / 151643;
- tokenizer: `Qwen2Tokenizer`, vocab base 151643, lunghezza 151665.

I checksum SHA-256 sono rimasti quelli validati nel Goal 8: `tokenizer_config.json` `c7462055...e5ddc4`, `special_tokens_map.json` `22bb30ae...dfdc`, `added_tokens.json` `58b54bbe...606b`, `vocab.json` `87a257b0...9e5d` e `merges.txt` `8831e4f1...904d5` (valori completi nel JSON riepilogativo).

## Cinque sessioni indipendenti e determinismo

Ogni sessione ha creato due container/processi nuovi, un process group nuovo, stage ricaricati e cache vuote. Le porte dedicate sono state 29508–29512. Il decoding è stato sempre `argmax(logits)`, senza sampling o temperatura, con `max_new_tokens=3`.

| sessione | prompt ids | token generati | testo completo | continuazione | EOS | arresto |
|---:|---|---|---|---|---|---|
| 1 | `[9707]` | `[11, 323, 10565]` | `Hello, and welcome` | `, and welcome` | no | `max_new_tokens` |
| 2 | `[9707]` | `[11, 323, 10565]` | `Hello, and welcome` | `, and welcome` | no | `max_new_tokens` |
| 3 | `[9707]` | `[11, 323, 10565]` | `Hello, and welcome` | `, and welcome` | no | `max_new_tokens` |
| 4 | `[9707]` | `[11, 323, 10565]` | `Hello, and welcome` | `, and welcome` | no | `max_new_tokens` |
| 5 | `[9707]` | `[11, 323, 10565]` | `Hello, and welcome` | `, and welcome` | no | `max_new_tokens` |

Il risultato greedy è quindi deterministico e ripetibile fra processi completamente nuovi: 5/5 sequenze identiche su entrambi i rank.

## Cache autoregressive

Il prefill è stato eseguito una sola volta. Lo stato restituito da ciascun layer è stato riutilizzato direttamente allo step successivo; le cache non sono state azzerate all'interno della sessione. In ciascuna delle cinque sessioni, per entrambi i rank e tutti i 28 layer:

| evento | posizione uniforme | byte cache per rank | crescita |
|---|---:|---:|---:|
| dopo prefill | 1 | 12.873.728 | — |
| dopo elaborazione token 1 | 2 | 12.902.400 | 28.672 |
| dopo elaborazione token 2 | 3 | 12.931.072 | 28.672 |

Non sono emerse divergenze di posizione fra layer. Il terzo token finale è stato restituito e decodificato, ma non è stato ripassato nel modello.

## Tempi

I valori sono millisecondi. `token 1` misura dal prefill Atlas fino alla ricezione del primo token; `token 2/3` includono il decode Atlas precedente, send Gloo e attesa del token corrente. Il totale è la generazione dal prefill al terzo token e non include il caricamento tokenizer/stage.

| sess. | tokenizz. | prefill Atlas | Gloo send/recv | prefill Argo | primo lm_head | token 1 | token 2 | token 3 | totale |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0,723 | 38.043,605 | 3,737 / 0,589 | 54.910,997 | 11.710,729 | 104.697,833 | 12.375,920 | 12.300,512 | 129.380,260 |
| 2 | 0,303 | 5.710,184 | 0,642 / 0,546 | 7.265,141 | 512,743 | 13.511,733 | 13.130,151 | 13.027,165 | 39.671,039 |
| 3 | 0,294 | 6.252,666 | 0,619 / 0,538 | 7.433,958 | 510,738 | 14.200,943 | 12.970,140 | 12.623,425 | 39.796,368 |
| 4 | 0,288 | 6.623,680 | 0,647 / 0,615 | 7.040,215 | 509,074 | 14.196,048 | 12.954,145 | 12.383,389 | 39.535,726 |
| 5 | 0,280 | 5.361,733 | 0,669 / 0,477 | 6.928,440 | 511,494 | 12.824,657 | 12.537,058 | 13.438,928 | 38.802,557 |

La sessione 1 è un outlier di cold start (prefill e primo `lm_head`), ma è funzionalmente corretta. La mediana degli s/token calcolata sui cinque totali di sessione è **13,223680 s/token**; min **12,934186**, max **43,126753**. Considerando le 15 latenze token individuali, mediana **12,970140 s**, min **12,300512**, max **104,697833**. Il tempo totale mediano è **39.671,039 ms**, min 38.802,557 e max 129.380,260 ms. OpenBLAS è la baseline; le prestazioni MKL non sono state usate come criterio di successo.

## RAM e swap

| sess. | atlas RSS/HWM MiB | atlas MemFree/Available MiB | argo RSS/HWM MiB | argo MemFree/Available MiB | swap processi |
|---:|---:|---:|---:|---:|---:|
| 1 | 13.208,56 / 13.223,64 | 14.447,50 / 28.806,81 | 15.164,04 / 15.164,04 | 14.760,16 / 30.334,12 | 0 / 0 MiB |
| 2 | 13.213,96 / 13.227,55 | 14.463,09 / 28.824,63 | 15.153,03 / 15.164,64 | 14.749,67 / 30.324,66 | 0 / 0 MiB |
| 3 | 13.226,26 / 13.226,26 | 14.474,05 / 28.836,12 | 15.164,56 / 15.164,56 | 14.740,20 / 30.315,39 | 0 / 0 MiB |
| 4 | 13.228,47 / 13.228,47 | 14.392,14 / 28.754,70 | 15.164,52 / 15.164,52 | 14.752,60 / 30.328,00 | 0 / 0 MiB |
| 5 | 13.226,43 / 13.226,43 | 14.448,18 / 28.835,45 | 15.164,94 / 15.164,94 | 14.762,24 / 30.337,85 | 0 / 0 MiB |

Su argo3 l'RSS per token mostra soltanto la piccola crescita/assestamento attesa delle cache (massimo circa 17 MiB nella sessione 1), senza crescita progressiva anomala. Tutti i campioni di swap di processo, inclusi quelli per token su argo3, sono zero.

## Kernel e shutdown

Il timestamp iniziale registrato su argo3 è `2026-08-15T12:45:09.707607723+02:00`; il controllo finale è stato eseguito a `2026-08-15T12:53:12.728610424+02:00`. Il journal kernel nel periodo riporta esplicitamente:

- nuovi SIGILL: **0**;
- nuovi `invalid opcode`: **0**;
- nuovi trap in `libtorch_cpu.so`: **0**;
- totale righe corrispondenti a `invalid opcode|SIGILL|trap|libtorch_cpu.so`: **0**.

Dopo ciascuna sessione entrambi i rank hanno riportato `clean_shutdown=true`; i container `--rm` sono scomparsi, la relativa porta non aveva listener e il gate ha consentito l'avvio della sessione successiva. Il controllo conclusivo non ha trovato container Goal 8R, listener sulle porte 29508–29512 o processi generativi residui.

## Differenze rispetto al Goal 8 originale

Il Goal 8 originale aveva completato la prima sessione con gli stessi token `[11, 323, 10565]` e lo stesso testo, ma la seconda sessione era terminata con exit 132 e un `invalid opcode` in `libtorch_cpu.so`. Goal 8R, con l'immagine OpenBLAS Ivy Bridge validata, completa invece cinque sessioni indipendenti consecutive senza crash e dimostra la ripetibilità greedy. La mediana attuale è circa 13,22 s/token contro circa 10,28 s/token della sola sessione riuscita del vecchio runtime; questo confronto è informativo e non è un criterio di successo. Pesi, tokenizer, prompt, partizione e protocollo autoregressivo sono invariati.

I dati completi machine-readable sono in `goal8r_generation_summary.json`; i JSON grezzi per rank sono `goal8r_s1_atlas.json` … `goal8r_s5_argo.json` con i corrispondenti stderr vuoti.
