# Publication audit

## Decision

**GO-PUBLISH-READY**. The local candidate tree contains no detected secrets,
model weights, prohibited binary artifacts, or unexplained large files. The
public code is parameterized, tests pass in the frozen Docker runtime, required
attribution and AI disclosure are present, and the Git dry run identifies only
the intended candidate files.

This decision describes the local publication candidate. No GitHub repository,
remote, commit, tag, release, push, or real staging operation was created.

## 1. Candidate inventory

Final candidate set: **63 files**, **357541 bytes** total.

| class | count | contents |
|---|---:|---|
| code | 22 | 6 files under `src/`; 16 tools/entry points under `scripts/` |
| tests | 5 | CPU ops, blocks, selective loader, partition, protocol |
| Docker/runtime | 5 | two Dockerfiles, two requirement locks, `pyproject.toml` |
| experimental reports | 17 | initial plan, Goal 2–9 reports, four Goal 8.5 summaries |
| JSON summaries | 4 | Goals 6, 8, 8R, and 9 aggregate results |
| public documentation | 8 | README, license, notices, acknowledgements, reproducibility, history, issue draft, this audit |
| ignore policy | 2 | `.gitignore`, `.dockerignore` |

The inventory was derived from
`git ls-files --cached --others --exclude-standard`, which is also the set
examined by the publication scans.

## 2. Secret and privacy audit

A content scan of every candidate checked for GitHub token formats, API/secret
key assignments, bearer credentials, private-key headers, password
assignments, AWS access-key formats, email addresses, cookie headers, and
common environment-secret forms. Result: **zero candidate matches and no real
secret found**.

The report intentionally does not print possible secret values. No personal
email is present in package metadata or documentation.

## 3. Local paths and network data

Nine historical Markdown reports contained an absolute personal home prefix or
an account-qualified SSH target. They were mechanically normalized to
`$HOME/...` and `<USER>@...`. This did not change benchmark values, stack
traces, model results, or private-IP evidence.

Historical private `192.168.x.x` addresses remain in engineering reports where
they describe the actual experiment. A separate scan of `src/`, `scripts/`,
and `tests/` found **zero hard-coded LAN IP addresses**. Both distributed CLI
drivers require an explicit `--master-addr`; README examples use
`<ATLAS_IP>`, `<ARGO_IP>`, `<MODEL_PATH>`, and `<TOKENIZER_PATH>`.

The experimental host names `atlas5` and `argo3` remain by design.

## 4. Weights, checkpoints, and large files

No Safetensors shard, checkpoint, copied tokenizer, PyTorch binary/wheel,
Docker image/archive, ELF dump, or core dump is present in the candidate set or
repository working tree. Searches covered `*.safetensors`, `*.bin`, `*.pt`,
`*.pth`, `*.ckpt`, `*.gguf`, `*.whl`, and core naming patterns.

No file in the repository exceeds 10 MiB. The largest candidate is a small
text/code artifact, so no large-file exception is required.

`.gitignore` and `.dockerignore` now exclude model/checkpoint/tokenizer
directories, weight formats, wheels, archives, core dumps, Python caches, raw
logs, stderr captures, and per-rank experiment JSON.

## 5. License and upstream attribution

- Original repository code/documentation: MIT License, copyright 2026 Sergio
  (Uraroga).
- Package metadata declares `license = "MIT"` and identifies only the human
  author, without an email.
- `THIRD_PARTY_NOTICES.md` attributes BICLab / SpikingBrain-7B and the tested
  Abel2076/SpikingBrain-7B-W8ASpike checkpoint.
- The notices state that this is an independent, unofficial community project,
  that upstream terms remain applicable, and that users must obtain model and
  tokenizer files separately.
- The repository MIT license does not claim to relicense upstream models or
  runtime dependencies.

No unresolved license/attribution blocker was identified from the material in
scope. Users still need to review upstream terms before downloading or using
the separately distributed model.

## 6. AI-assisted development disclosure

`ACKNOWLEDGEMENTS.md` contains the requested substantial-assistance disclosure
for ChatGPT and OpenAI Codex. It distinguishes their roles from human project
responsibility and states explicitly that they are not authors/co-authors,
OpenAI is not a partner, and no affiliation, sponsorship, or endorsement is
claimed. No OpenAI or ChatGPT logo is used.

## 7. Public documentation and consistency

Created:

1. `README.md`
2. `LICENSE`
3. `ACKNOWLEDGEMENTS.md`
4. `THIRD_PARTY_NOTICES.md`
5. `REPRODUCIBILITY.md`
6. `EXPERIMENT_HISTORY.md`
7. `UPSTREAM_ISSUE_DRAFT.md`
8. `PUBLICATION_AUDIT.md`

Updated:

- `pyproject.toml` with description, README, MIT license, and human author;
- `.gitignore` and `.dockerignore` with publication exclusions;
- `PROGETTO_CLUSTER_CPU.md` with a visible historical-plan notice;
- nine historical reports only to normalize personal paths.

README, reproducibility, and history agree on 7,692,495,104 parameters (~7.69B),
28 layers, hidden size 3584, FP32 checkpoint storage, 14/14 split, atlas layers
0–13, argo layers 14–27 plus norm/head, PyTorch
`2.13.0+openblas.ivybridge`, and the stable OpenBLAS runtime. Goal 8R and Goal 9
values match their aggregate JSON reports.

The initial 16/12 proposal remains intact in `PROGETTO_CLUSTER_CPU.md`, now
clearly labeled as a historical plan superseded by Goal 3 block benchmarks.
The SIGILL failure and failed environment mitigations remain visible rather
than being rewritten as an uninterrupted success story.

The Markdown link audit found zero missing local targets. Heading, table, code
fence, obvious typo, and trailing-whitespace checks found no blocking issue.

## 8. Tests and CLI portability

Tests ran with the public workspace mounted read-only into the frozen image:

```text
spikingbrain-cpu:goal8.6-openblas
sha256:696d98683cee6eef018a73e5c70b402f9103fc24b230356975a7ef1685aed836
41 passed in 3.07 s
```

The only warning was `PytestCacheWarning` because pytest could not write its
cache into the intentionally read-only workspace. Package import succeeded.
No real weights were loaded.

These commands also exited successfully:

```text
python scripts/distributed_generate.py --help
python scripts/distributed_stage.py --help
```

Both show required `--master-addr` and `--peer` parameters. No `192.168.*`
literal exists under `src/`, `scripts/`, or `tests/`.

## 9. Ignored local files

There are **200 existing ignored files**, retained locally and not deleted:

| ignored group | existing count |
|---|---:|
| raw logs and stress directories | 124 |
| per-rank/temporary JSON | 31 |
| stderr captures (`*.err`) | 31 |
| Python/pytest caches | 9 |
| raw diagnostic text captures | 5 |

Relevant ignore rules include `.venv/`, `__pycache__/`, `.pytest_cache/`,
`*.pyc`, `*.err`, `*.log`, `goal85_logs/`, `goal86_logs/`, per-rank Goal
7/8/8R/9 JSON, raw diagnostic captures, all named model-weight extensions,
`models/`, `checkpoints/`, `Modelli/`, and `tokenizer/`.

## 10. Open issues

- The project remains experimental and tested only on atlas5/argo3.
- The stable Docker image is identified but not published by this audit; users
  may need to build it from the pinned Dockerfile.
- Model and tokenizer licenses/terms must be reviewed at their upstream source
  before separate download or use.
- The draft upstream issue must not be posted until the human author approves
  it.
- Performance remains limited by per-forward fake quantization of FP32 weights.
- No formal model-quality evaluation or external reproduction has been done.

None of these is a secret, weight leak, internal contradiction, or test failure
that blocks publication of the documented experimental repository.

## 11. Final Git dry run

The permitted checks were run after the final content audit:

```text
git status --short
git diff --check
git add -n .
```

`git diff --check` reported no whitespace errors. `git add -n .` proposed
exactly **63 candidate files** and did not stage them. `git status --short`
continued to show the publication candidates as untracked; no index mutation
was performed.

`git remote -v` is empty. No remote was added, no commit/tag/release was
created, and no push or real `git add` was performed. Goal 11 was not started.
