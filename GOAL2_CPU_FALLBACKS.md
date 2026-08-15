# Goal 2 — Fallback CPU di SpikingBrain

## Decisione

**GO — possiamo passare al loader selettivo e alla costruzione delle partizioni**, limitatamente a smoke test con contesti brevi e con misurazione rigorosa della RAM. I fallback fondamentali funzionano su CPU con dati sintetici e senza Triton, `fla`, FlashAttention o CUDA.

Questo GO non equivale a una previsione di inferenza interattiva. L'attention di riferimento è quadratica e QuantLinear continua a fake-quantizzare pesi FP32 a ogni chiamata. Entrambi richiederanno ottimizzazione dopo la prima validazione end-to-end.

## Implementazione

Il codice è autonomo in `src/spikingbrain_cpu/ops.py` e non importa alcun modulo dal checkpoint.

| Operatore CPU | Sostituisce | Note |
|---|---|---|
| `RMSNorm` | `fla.modules.RMSNorm` | Riduzione FP32 per input BF16/FP16; supporta anche il percorso fused residual/prenorm usato dal blocco MLP. |
| `swish` | `fla.modules.activations.swish` | Alias funzionale a `torch.nn.functional.silu`. |
| `swiglu` | `activations.swiglu_linear` | `silu(gate) * value`, con down-projection opzionale tramite `F.linear`. |
| `sliding_window_attention` | `flash_attn_func` e `flash_attn_varlen_func` | CPU, causale, GQA, mask opzionale e semantica FlashAttention `(left, 0)`. Supporta query corte allineate a destra per cache decoding. |
| `gla_recurrent` | `chunk_gla`, `fused_chunk_gla`, `fused_recurrent_gla` | Ricorrenza sequenziale leggibile, stato iniziale/finale e accumulo FP32 per BF16/FP16. |
| `dynamic_spikes` | `quant_linear.dynamic_spikes` | Replica threshold, rounding e ricostruzione. Il bitwise encode/decode di `neuron.py` non viene materializzato perché senza clipping restituisce gli stessi interi. |
| `fake_quantize_weight`, `quant_linear` | percorso di inferenza di `QuantLinear` | Replica fake quantizzazione a gruppi e activation spikes; offre un buffer riutilizzabile senza modificare il peso. |

`ShortConvolution` non è stato implementato: `config.json` imposta `use_short_conv: false`, quindi non è percorso dall'architettura di questo checkpoint. Implementarlo ora non aggiungerebbe copertura al modello target.

### Formula GLA

Per ogni token, testa e batch:

```text
S_t = exp(g_t) * S_(t-1) + k_t outer v_t
o_t = (q_t / sqrt(d_k)) @ S_t
```

`g_t` corrisponde al `logsigmoid(gk) / 16` prodotto dal codice del checkpoint. Stato e output sono stati validati sia con un caso scalare calcolabile a mano, sia confrontando esecuzione unica e prefill+decode con stato cache. La formula e lo scaling coincidono con il percorso recurrent di FLA ispezionato come riferimento; non è stato installato né eseguito `fla`.

## Test

Comando:

```bash
.venv/bin/python -m pytest -q
```

Risultato su atlas5: **15 passed in 1,25 s**.

Copertura verificata:

- shape e dtype FP32/BF16;
- RMSNorm e SwiGLU contro formulazioni PyTorch equivalenti;
- somma fused residual/prenorm di RMSNorm;
- causalità: una modifica futura non cambia output passati;
- confine esatto della sliding window e GQA con KV heads ridotte;
- righe interamente mascherate senza NaN;
- GLA scalare con risultati attesi `[1, 1.5, 1.75]`;
- equivalenza GLA tra passaggio unico e passaggi con cache;
- determinismo di attention e GLA;
- assenza di NaN/Inf;
- equivalenza QuantLinear fra formula allocante, buffer riusabile e formula esplicita;
- immutabilità dei pesi sintetici originali.

PyTorch emette un warning non funzionale perché NumPy non è installato nel nuovo ambiente; i test non usano NumPy e il warning non altera i risultati.

## Benchmark atlas5

Ambiente: Intel i5-4590, PyTorch `2.13.0+cpu`, Python 3.12.3, 4 thread, `cuda_available=false`. Ogni valore è la mediana di 20 iterazioni dopo 5 warmup, dentro `torch.inference_mode()`.

Comando:

```bash
.venv/bin/python scripts/benchmark_cpu_ops.py --iterations 20 --warmup 5 --threads 4
```

| Operazione sintetica | Mediana | Min–max |
|---|---:|---:|
| RMSNorm `[1,32,3584]` | **0,073 ms** | 0,070–0,082 ms |
| core SwiGLU `[1,8,18944]`, senza down projection | **0,074 ms** | 0,072–0,202 ms |
| attention prefill `[1,28,64,128]`, 4 KV heads, window 64 | **1,618 ms** | 1,439–7,156 ms |
| attention decode: query 1, cache 4096, 28/4 heads, d=128 | **33,652 ms** | 29,126–36,430 ms |
| GLA recurrent `[1,28,32,128]` | **37,985 ms** | 33,101–67,803 ms |
| GLA decode 1 token con stato `[1,28,128,128]` | **1,786 ms** | 1,622–4,488 ms |
| QuantLinear allocante, peso 1024×1024 | **3,041 ms** | 2,604–12,285 ms |
| QuantLinear con buffer, peso 1024×1024 | **0,613 ms** | 0,518–9,249 ms |

Sono benchmark dei soli operatori: non includono tutte le proiezioni 3584→18944, norm, residual, cache management o rete. Non devono essere sommati come stima completa del tempo/token.

### argo3

Il Python predefinito di argo3 non ha PyTorch (`ModuleNotFoundError: torch`) e non è presente un virtualenv alternativo rilevato. Per ripetere il benchmark sarebbe necessario installare PyTorch CPU su argo3 oppure copiarvi il piccolo progetto/virtualenv compatibile. In osservanza del vincolo esplicito, **nessun file è stato copiato e nessun pacchetto è stato installato su argo3**. Il benchmark remoto resta quindi pendente.

## QuantLinear e RAM

Il checkpoint non contiene pesi compressi: `Quantizer.forward` calcola:

```python
(weight.reshape(...)/scales).round()*scales
```

Il profiler PyTorch con un peso sintetico 1024×1024 FP32 (4 MiB) rileva:

- percorso originale/allocante: **12,02 MiB** di traffico cumulativo dell'allocator, dovuto principalmente ai risultati di `div`, `round` e `mul`;
- percorso con buffer creato prima del forward: **0,02 MiB** nel forward profilato;
- la fake-quantizzazione allocante mantiene almeno due risultati grandi quanto la matrice contemporaneamente in alcuni punti dell'espressione;
- il risultato passato a `F.linear` è comunque un secondo peso denso FP32: nessuna riduzione di storage o GEMM int8.

La matrice più grande assunta dal checkpoint, 18944×3584 FP32, occupa circa **259 MiB**. Ne segue:

- espressione allocante: picco extra stimato fino ad almeno **~518 MiB** per due temporanei sovrapposti, oltre a churn cumulativo di circa 777 MiB per chiamata;
- buffer riutilizzabile: **259 MiB extra persistenti**, ma niente grandi allocazioni per `div/round/mul` durante ciascun forward;
- il buffer può essere uno per processo, dimensionato alla matrice locale massima e riutilizzato sequenzialmente;
- non deve essere creato un buffer per ciascun layer;
- non va fake-quantizzato l'intero state dict in una volta.

Il buffer riduce RAM di picco e tempo nel test sintetico, ma rifare divisione/round/moltiplicazione su miliardi di pesi a ogni token resta un forte collo di bottiglia di banda memoria. Una cache permanente dei pesi fake-quantizzati raddoppierebbe quasi lo storage dei pesi e non è compatibile con l'attuale budget; una vera conversione int8 richiederebbe una strategia diversa ed è fuori Goal 2.

## Limiti numerici e prestazionali

- RMSNorm, softmax attention e stato GLA usano accumulo FP32 per input BF16/FP16 e restituiscono il dtype di input.
- L'attention restituisce zero, non NaN, se una query non ha chiavi valide.
- L'attention di riferimento materializza score `[batch, heads, query, key]`: la mask sliding-window garantisce correttezza ma non riduce la complessità quadratica del prefill. Non è utilizzabile al context massimo 131072 e richiederà un'implementazione a blocchi/bande.
- In decoding con cache 4096 il solo core attention costa già ~33,7 ms per layer attention su atlas5; argo3 sarà probabilmente più lento.
- La GLA sequenziale è una baseline corretta, non un kernel ottimizzato. Il decoding a un token è ragionevole; il prefill lungo soffrirà il loop Python.
- Non è disponibile un confronto numerico diretto con i kernel GPU originali nello stesso ambiente. I test verificano matematica, cache e invarianti, non equivalenza bit-a-bit a Triton/FlashAttention.

## Cosa manca per AtlasStage e ArgoStage

Prima di collegare i pesi reali servono ancora:

1. loader Safetensors selettivo per prefisso/range, con costruzione dei moduli su `meta` e materializzazione tensor-by-tensor;
2. wrapper dei blocchi con proiezioni Q/K/V/GK/O e MLP, residual e norm, che usino esclusivamente questi operatori;
3. rotary embeddings CPU per i layer sliding-window;
4. cache per-layer che separi stato GLA e KV sliding-window e rispetti gli indici globali 0..27;
5. mapping rigoroso di tutti i nomi/shape del checkpoint verso i due stage;
6. buffer QuantLinear unico e riusabile per processo, con misura del vero picco RSS;
7. test di equivalenza del blocco con pesi sintetici, poi smoke test locale a due stage;
8. protocollo Gloo e gestione coerente di mask, `position_ids`, `cache_position`, sampling ed EOS;
9. benchmark di argo3 dopo autorizzazione a predisporre un ambiente PyTorch CPU;
10. attention banded/chunked e GLA prefill ottimizzata prima di tentare contesti non banali.

## Integrità e vincoli

- Nessuno shard Safetensors è stato aperto o caricato.
- Non è stato chiamato `from_pretrained()` e non è stato creato il modello completo.
- La directory `$HOME/Modelli/SpikingBrain` non è stata creata su atlas5.
- Nessun peso o file del progetto è stato copiato su argo3.
- I file originali del checkpoint sono stati soltanto letti via SSH e non modificati.
- L'unica installazione è `.venv` locale nel progetto su atlas5 con PyTorch CPU e pytest; `.venv` è esclusa dal versionamento.
