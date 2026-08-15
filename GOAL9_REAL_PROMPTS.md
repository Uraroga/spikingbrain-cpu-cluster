# Goal 9 — Collaudo con prompt reali brevi

## Decisione finale

**GO-PROMPTS**: le quattro sessioni valide e indipendenti sono complete; hidden e logits sono finiti a ogni step, tutti i token sono nel range del tokenizer, le cache sono coerenti sui 28 layer, lo swap di processo è sempre zero, il journal riporta 0 SIGILL/invalid opcode/trap e lo shutdown finale è pulito.

Non sono stati modificati runtime, OpenBLAS, pesi, partizione 14/14 o backend Gloo. Goal 10 non è stato avviato.

## Funzionamento tecnico

### Baseline e preflight

Entrambi i nodi hanno usato esclusivamente `spikingbrain-cpu:goal8.6-openblas`, Image ID identico all'atteso:

`sha256:696d98683cee6eef018a73e5c70b402f9103fc24b230356975a7ef1685aed836`

Su atlas5 e argo3: PyTorch `2.13.0+openblas.ivybridge`, MKL `False`, MKLDNN `False`, CUDA `False`. La capability su argo3 è `DEFAULT`. Gloo ha usato `eno1` con `--network host`; checkpoint, tokenizer e driver sono stati montati read-only. Ogni sessione valida ha avviato processi, process group, stage e cache nuovi, con quattro thread per rank.

### Chat template reale

`tokenizer_config.json` contiene realmente un `chat_template` (SHA-256 del file `c7462055715b3b866c9761ad749bfb364af26eb3776e07d20d14f98509e5ddc4`). Senza un messaggio system esplicito, il template aggiunge:

- `<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n`;
- il messaggio user racchiuso tra `<|im_start|>user\n` e `<|im_end|>\n`;
- `<|im_start|>assistant\n` come generation prompt.

La sessione chat ha chiamato direttamente `tokenizer.apply_chat_template(..., add_generation_prompt=True)`. Non è stata costruita manualmente alcuna stringa ChatML e il template non è stato modificato.

### Tokenizzazione e output esatti

| sessione | prompt/messaggio esatto | token prompt | n | special token nel prompt | token generati | continuazione esatta | EOS/arresto |
|---|---|---|---:|---|---|---|---|
| A | `The capital of France is` | `[785, 6722, 315, 9625, 374]` | 5 | nessuno | `[12095, 13, 576, 6722, 315]` | ` Paris. The capital of` | no / max 5 |
| B | `Il cielo è` | `[12050, 11825, 20172, 11422]` | 4 | nessuno | `[650, 6, 12543, 23392, 1853]` | ` un'immagine di` | no / max 5 |
| C | `Hello, how are you?` | `[9707, 11, 1246, 525, 498, 30]` | 6 | nessuno | `[358, 1079, 6915, 11, 9702]` | ` I am fine, thank` | no / max 5 |
| chat | `Reply briefly: What is 2 + 2?` | 30 token, lista sotto | 30 | `<|im_start|>` ×3, `<|im_end|>` ×2 | `[20841, 26753, 25, 220, 19, 13, 151643]` | `Reply briefly: 4.<|endoftext|>` | sì / EOS al token 7 |

Token prompt chat:

`[151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 20841, 26753, 25, 3555, 374, 220, 17, 488, 220, 17, 30, 151645, 198, 151644, 77091, 198]`

Testi completi esatti:

- A: `The capital of France is Paris. The capital of`
- B: `Il cielo è un'immagine di`
- C: `Hello, how are you? I am fine, thank`
- chat: `<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nReply briefly: What is 2 + 2?<|im_end|>\n<|im_start|>assistant\nReply briefly: 4.<|endoftext|>`

Nel testo chat sopra, `\n` rappresenta il carattere newline realmente prodotto; il JSON conserva la stringa completa machine-readable.

### Cache

Le posizioni sotto sono identiche su entrambi i rank e uniformi su tutti i 28 layer. La crescita è sempre 28.672 byte per token ripassato nel modello e per rank.

| sessione | posizione dopo prefill | posizioni successive | byte dopo prefill | byte finali | divergenze |
|---|---:|---|---:|---:|---:|
| A | 5 | 6, 7, 8, 9 | 12.988.416 | 13.103.104 | 0 |
| B | 4 | 5, 6, 7, 8 | 12.959.744 | 13.074.432 | 0 |
| C | 6 | 7, 8, 9, 10 | 13.017.088 | 13.131.776 | 0 |
| chat | 30 | 31, 32, 33, 34, 35, 36 | 13.705.216 | 13.877.248 | 0 |

I token finali che causano `max_new_tokens`, e l'EOS finale della chat, non sono ripassati nel modello.

### Controllo numerico

Per ogni step e sessione sono stati verificati esplicitamente hidden Atlas, hidden ricevuti e prodotti da Argo, logits e token ID. Tutti gli hidden/logits sono finiti, senza NaN o Inf. Tutti i token restituiti sono compresi in `[0, len(tokenizer))`.

### Tempi

Valori in millisecondi. Il tempo del token 1 include il prefill distribuito; gli altri sono decode autoregressivi. La velocità non è stata usata come criterio GO/STOP.

| sess. | tokenizz. | prefill Atlas | prefill Argo | primo lm_head | tempi token | mediana token | totale | s/token |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| A | 0,458 | 7.250,123 | 7.924,836 | 510,654 | 15.690,287; 13.831,763; 13.686,954; 14.250,961; 13.819,889 | 13.831,763 | 71.283,017 | 14,256603 |
| B | 0,334 | 7.071,001 | 8.024,427 | 511,667 | 15.612,127; 13.636,969; 13.427,071; 13.570,462; 13.414,241 | 13.570,462 | 69.663,474 | 13,932695 |
| C | 0,363 | 7.573,207 | 8.424,633 | 513,285 | 16.514,460; 13.583,054; 13.102,434; 13.528,456; 13.655,777 | 13.583,054 | 70.386,631 | 14,077326 |
| chat | 17,003 | 11.380,807 | 10.727,053 | 514,085 | 22.629,099; 15.800,368; 13.560,476; 12.664,260; 13.064,862; 12.839,969; 12.819,156 | 13.064,862 | 103.381,972 | 14,768853 |

Il primo token di ciascuna sessione è il tratto più lento perché comprende il prefill; il chat prefill è più lungo (30 token). Non è emerso un ulteriore cold-start anomalo come l'outlier della prima sessione Goal 8R.

### RAM e swap

| sess. | atlas RSS/HWM MiB | atlas MemFree/Available MiB | argo RSS/HWM MiB | argo MemFree/Available MiB | swap atlas/argo |
|---|---:|---:|---:|---:|---:|
| A | 13.227,23 / 13.227,23 | 14.175,02 / 28.576,74 | 15.163,12 / 15.164,37 | 14.758,53 / 30.338,23 | 0 / 0 |
| B | 13.217,84 / 13.236,30 | 14.195,26 / 28.599,25 | 15.171,45 / 15.171,45 | 14.758,99 / 30.338,90 | 0 / 0 |
| C | 13.230,12 / 13.230,12 | 14.238,86 / 28.645,10 | 15.166,93 / 15.166,93 | 14.761,36 / 30.341,57 | 0 / 0 |
| chat | 13.237,82 / 13.237,82 | 13.614,51 / 28.398,50 | 15.169,45 / 15.172,68 | 14.739,94 / 30.324,85 | 0 / 0 |

I campioni RSS per step su argo3 oscillano entro circa 16 MiB senza crescita progressiva anomala. Tutti i campioni di swap di processo sono zero.

### Kernel, shutdown e nota sul guard del prompt chat

Intervallo journal argo3: da `2026-08-15T13:03:31.304665436+02:00` a `2026-08-15T13:18:32.633951579+02:00`:

- nuovi SIGILL: **0**;
- nuovi `invalid opcode`: **0**;
- nuovi trap `libtorch_cpu.so`: **0**;
- totale righe corrispondenti: **0**.

Il primo avvio chat non è contato fra le sessioni generative valide: il guard storico del ricevitore Gloo accettava al massimo 8 token di prefill, mentre il template reale ne produce 30. Argo ha restituito un normale `ValueError: invalid hidden length 30`, senza SIGILL o crash nativo; atlas è stato chiuso mentre attendeva il peer. Il guard è stato corretto per usare il parametro esplicito `max_prompt_tokens` (64 nel solo test chat), quindi la sessione è ripartita integralmente con processi/cache/process group nuovi sulla porta 29524 ed è riuscita. Questo modifica soltanto la validazione del protocollo applicativo, non runtime, modello o tokenizer.

Dopo ogni sessione valida entrambi i rank hanno riportato `clean_shutdown=true`. Il controllo conclusivo non ha trovato container Goal 9, listener sulle porte 29520–29524 o processi generativi residui.

## Qualità osservata

Queste osservazioni sono descrittive e non costituiscono una valutazione generale della qualità del modello:

- A è grammaticalmente plausibile e pertinente (`Paris`), ma ripete l'inizio della frase e resta incompleta al limite di cinque token.
- B è grammaticalmente plausibile e semanticamente possibile, ma resta incompleta al limite di cinque token.
- C è grammaticalmente plausibile e pertinente alla domanda, ma resta incompleta al limite di cinque token.
- La chat risponde correttamente e brevemente con `4`, ripete parte dell'istruzione (`Reply briefly:`) e termina con EOS reale al settimo token.

Non emergono output tanto poveri o incoerenti da richiedere `GO-TECHNICAL-QUALITY-QUESTIONABLE`; la decisione resta strettamente tecnica: **GO-PROMPTS**.
