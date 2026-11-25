# 📊 QuerentCrew: a multi-agent analytics assistant

*Querent* (KWEER-ent): the person who asks a question, from the Latin *quaerere*, "to ask, to seek".
You are the querent. A **crew** of AI agents (planner, SQL writer, SQL guard, executor, analyst,
chart builder, reporter) works on your question and returns an answer backed by real numbers.

> Ask business questions in plain English and get a short report backed by numbers.
> A team of AI agents built with **LangChain 1.x** and **LangGraph** plans the analysis, writes safe SQL,
> fixes its own errors, crunches results with **Pandas/NumPy**, draws charts, and pauses for **human
> approval** before touching personal data. Every prompt is **versioned and measured** by an
> evaluation harness.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue) ![LangChain](https://img.shields.io/badge/LangChain-1.x-green)
![LangGraph](https://img.shields.io/badge/LangGraph-1.x-purple) ![Tests](https://img.shields.io/badge/tests-pytest-orange)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED) ![AWS](https://img.shields.io/badge/deploy-AWS%20EC2-FF9900)

![QuerentCrew home screen](docs/screenshot_home.png)

![An answer with the interactive chart, stats and tabs](docs/screenshot_answer.png)

---

## Table of contents

1. [The problem](#1-the-problem)
2. [What it does (demo walkthrough)](#2-what-it-does-demo-walkthrough)
3. [Features](#3-features)
4. [Architecture](#4-architecture)
5. [The agents](#5-the-agents)
6. [Safety and governance](#6-safety-and-governance)
7. [Prompt engineering and versioning](#7-prompt-engineering-and-versioning)
8. [Evaluation](#8-evaluation)
9. [Tech stack](#9-tech-stack)
10. [Project structure](#10-project-structure)
11. [Getting started](#11-getting-started)
12. [Usage: UI, CLI, REST API](#12-usage)
13. [Configuration](#13-configuration)
14. [Testing](#14-testing)
15. [Deployment (Docker, AWS, Streamlit Cloud)](#15-deployment-docker-aws-streamlit-cloud)
16. [Design decisions](#16-design-decisions)
17. [Limitations and roadmap](#17-limitations-and-roadmap)

---

## 1. The problem

Business teams have questions like *"Why did revenue drop in the South last quarter?"*. Answering one
usually means waiting for an analyst to write SQL, export to a spreadsheet, build a chart and write
a summary. Plain "chat with your database" demos send one LLM-written query to the database and hope
for the best. They break on multi-step questions, can run dangerous SQL, leak personal data and
invent numbers.

**QuerentCrew treats the problem as an engineered workflow, not a single prompt:** planning,
guarded SQL generation with self-correction, deterministic analysis, human oversight and measured
prompt quality.

## 2. What it does (demo walkthrough)

The bundled sample database (`scripts/seed_database.py`) holds one year of sales for a fictional
company: 400 customers, about 3,200 orders and about 6,400 order lines across 4 regions, 4 product
categories, 3 segments and 3 channels. A **real business story is planted in the data**: South
region revenue falls sharply in Q3 2025 because Electronics demand and Enterprise orders drop.

**Question:** *"Why did revenue drop in the South region in Q3 2025?"*

| Step | Agent | What happens |
|---|---|---|
| 1 | **Router** | Classifies it as a `data_question` (vs. general or out-of-scope chat) |
| 2 | **Planner** | Splits it into sub-questions: (a) Q2 vs Q3 revenue by region, (b) Q2 vs Q3 South revenue by category, (c) … by customer segment |
| 3 | **SQL writer** | Writes SQLite for sub-question (a) using the schema + business glossary |
| 4 | **SQL guard** | Checks it's read-only, references real tables, adds a row LIMIT, detects PII |
| 5 | **Executor** | Runs it on a read-only connection with a timeout. If SQLite errors, the error goes back to the SQL writer, which fixes the query (up to 2 retries) |
| 6 | *loop* | Steps 3-5 repeat for each sub-question |
| 7 | **Analyst** | Pandas computes % changes, shares and peaks; the LLM turns those numbers into findings |
| 8 | **Charter** | Picks a chart type and columns (validated against the real columns; if the model declines or picks invalid columns, a heuristic chooses one). The UI draws it as an interactive Altair chart; Matplotlib renders a PNG for the API and CLI |
| 9 | **Reporter** | Writes the final Markdown answer: headline number, key findings, recommended next step |

Follow-up: *"Now show the same for the North region"*. Conversation memory (a LangGraph
checkpointer) lets the planner resolve "the same".

Personal data: *"List the emails of our top 5 Enterprise customers"*. The guard flags the `email`
column, the graph **pauses** for approval, and after approval the emails are still **masked** in the
output (`ro***@example.com`).

Sample chart generated from the bundled data:

![Revenue by region, Q2 vs Q3](docs/sample_chart.png)

## 3. Features

- **Multi-agent orchestration with LangGraph:** 11-node `StateGraph` with conditional routing, loops and a retry cycle
- **Planning / task decomposition:** complex questions become up to 3 SQL-sized steps
- **Self-correcting SQL:** database errors and guard rejections are fed back to the SQL agent
- **Human-in-the-loop:** LangGraph `interrupt()` pauses the run, `Command(resume=...)` continues it from the UI, CLI or API
- **Conversation memory:** per-thread state in a checkpointer; follow-up questions work
- **Structured output everywhere:** every agent returns a typed Pydantic model (`RouteDecision`, `AnalysisPlan`, `SQLDraft`, `Insights`, `ChartSpec`)
- **Deterministic analytics:** Pandas/NumPy calculate the numbers; the LLM explains them, so it doesn't invent figures
- **Defence-in-depth SQL safety:** rule-based guard, read-only SQLite URI, SQLite authorizer, query timeout, row cap
- **PII governance:** approval gate + automatic masking of personal columns
- **Versioned prompt registry:** YAML prompts, three SQL-writer versions (zero-shot → rules + glossary → few-shot + reasoning)
- **Evaluation harness:** execution accuracy on 40 gold questions (20 dev + 20 held-out, written after the prompts were frozen), 3 runs per question, a grader validated by mutation testing and checked on 5 extra databases and by manual review, 95% intervals and paired tests, free re-grading of the published answers, file fingerprints, leakage tests, resumable runs
- **Model-agnostic:** Groq, Gemini, OpenAI, Ollama (local), Anthropic or Bedrock via `init_chat_model`; automatic retry and optional fallback model
- **Observability:** token/latency tracking per agent, step trace in the UI, optional LangSmith tracing
- **Data integration:** upload any CSV; Pandas cleans headers and infers types, and it becomes a queryable table
- **Three interfaces:** Streamlit chat UI, FastAPI REST API (with OpenAPI docs), terminal CLI
- **Engineering:** OOP with dependency injection, 83 pytest tests that need **no API key**, Docker, Docker Compose, GitHub Actions CI, AWS deployment guide

## 4. Architecture

```mermaid
flowchart TD
    U([User question]) --> R{Router}
    R -- general / out of scope --> RESP[Responder] --> E([Answer])
    R -- data question --> P[Planner<br/>1-3 sub-questions]
    P --> W[SQL writer<br/>prompt v1/v2/v3]
    W --> G{SQL guard}
    G -- blocked & retries left --> W
    G -- needs approval: PII --> H[[Human approval<br/>interrupt / resume]]
    G -- ok --> X[SQL executor<br/>read-only + timeout]
    H -- approved --> X
    H -- rejected --> A
    G -- blocked & no retries --> A
    X -- SQLite error & retries left --> W
    X -- rows --> A[Advance]
    A -- more sub-questions --> W
    A -- done --> AN[Analyst<br/>Pandas stats + LLM]
    AN --> C[Charter<br/>Altair + Matplotlib]
    C --> REP[Reporter] --> E

    subgraph Memory
      CP[(LangGraph checkpointer<br/>thread_id → state)]
    end
    subgraph Data
      DB[(SQLite)]
      CSV[CSV upload] --> DB
    end
    X --- DB
```

**Layers**

| Layer | Module | Responsibility |
|---|---|---|
| Interfaces | `app/streamlit_app.py`, `app/style.css`, `api.py`, `__main__.py` | UI, REST, CLI |
| Facade | `service.py` → `QuerentCrew` | `ask()`, `resume()`, `history()`, `import_csv()` |
| Orchestration | `graph.py` | LangGraph wiring, checkpointer |
| Agents | `nodes.py` → `AgentNodes` | One method per agent + routing functions |
| LLM | `llm.py` | `LLMClient` interface, `LangChainLLM`, `UsageTracker` |
| Prompts | `prompts/*.yaml`, `registry.py` | Versioned templates |
| Safety | `sql_safety.py` | `SQLGuard`, PII masking |
| Data | `database.py`, `analysis.py`, `charts.py` | Read-only execution, Pandas profiling, charts |
| Quality | `evaluation.py`, `evals/` | Execution-accuracy benchmark |

## 5. The agents

| Agent | Input | Output (Pydantic) | Notes |
|---|---|---|---|
| Router | question + history | `RouteDecision` | Falls back to the data path if the LLM call fails |
| Responder | question + schema | text | Explains the data and suggests example questions |
| Planner | question + history + schema + glossary | `AnalysisPlan` | Capped at `MAX_SUB_QUESTIONS`; resolves follow-ups |
| SQL writer | sub-question + schema + glossary + feedback | `SQLDraft` | Prompt version selectable; gets error feedback on retries |
| SQL guard | SQL | `GuardResult` | Rule-based, no LLM: predictable and unit-tested |
| Human approval | guard result | decision | `interrupt()`; resumes with approve/reject + comment |
| SQL executor | SQL | rows | Read-only, authorizer, timeout, row cap, PII masking |
| Analyst | results + Pandas profile | `Insights` | Numbers come from `DataProfiler`, not the LLM |
| Charter | result overview | `ChartSpec` | Spec validated against real columns; heuristic fallback |
| Reporter | insights + results + notes | Markdown | Mentions caveats, failed or rejected steps |

## 6. Safety and governance

LLM-written SQL is untrusted input, so it passes through four independent layers:

1. **`SQLGuard` (static rules):** single statement only; must start with `SELECT`/`WITH`; blocks
   `INSERT/UPDATE/DELETE/DROP/ALTER/ATTACH/PRAGMA/...` (string literals ignored, so `WHERE status =
   'delete'` is fine); unknown tables rejected with a list of valid ones; CTE names understood;
   `LIMIT` added to non-aggregated queries.
2. **Read-only connection:** `file:...?mode=ro` SQLite URI.
3. **SQLite authorizer:** denies every action except read/select/function, even if layers 1-2 were bypassed (tested).
4. **Timeout + row cap:** a progress handler aborts long queries; results are capped at `MAX_RESULT_ROWS`.

**PII:** columns listed in `PII_COLUMNS` (default `email, phone`) trigger the approval gate
(`REQUIRE_APPROVAL_FOR_PII`) and are always masked in results. `REQUIRE_APPROVAL_FOR_ALL=true` turns
on approval for every query (useful for high-risk databases).

## 7. Prompt engineering and versioning

Prompts are data, not code: `src/querentcrew/prompts/<agent>.yaml`, each with named versions and a
description. They use `$variable` templates so SQL braces never need escaping.

| SQL writer version | Technique |
|---|---|
| `v1` | Zero-shot baseline: schema only |
| `v2` | Role prompting + explicit rules (date handling, rounding, aliases, PII) + **business glossary** (how revenue, AOV and return rate are defined) |
| `v3` | v2 + **few-shot examples** + brief step-by-step reasoning field + error-repair instructions |

The glossary (`data/business_glossary.md`) is domain context kept separate from the prompt, so a new
dataset only needs a new glossary. Switch versions with `PROMPT_VERSION_SQL=v2` or compare them all
in the evaluation harness.

## 8. Evaluation

The evaluation answers one question: **does each prompt version write SQL that returns the right
data?** It is built so that anyone can rerun it, re-grade it for free and check every verdict.

### Results at a glance

Model `groq:openai/gpt-oss-120b`, temperature 0, 40 questions × 3 prompt versions × 3 runs = 360
answers. Accuracy is the mean over the 3 runs.

| Suite | Prompt | Original grader (as recorded) | Corrected grader, checked on 6 databases | 95% interval | Manual review (ambiguous counted wrong – right) |
|---|---|---|---|---|---|
| dev | v1 zero-shot | 35.0% | 40.0% | 21.9–61.3% | 45.0% |
| dev | v2 + glossary and rules | 81.7% | 90.0% | 69.9–97.2% | 86.7–91.7% |
| dev | v3 + few-shot examples | 90.0% | 96.7% | 78.7–99.6% | 95.0–100.0% |
| holdout | v1 zero-shot | 50.0% | 60.0% | 38.7–78.1% | 63.3% |
| holdout | v2 + glossary and rules | 100.0% | 100.0% | 83.9–100.0% | 100.0% |
| holdout | v3 + few-shot examples | 100.0% | 100.0% | 83.9–100.0% | 100.0% |

What the numbers say:

- **The business glossary (v1 → v2) is the large, statistically clear gain.** Most v1 errors count
  returned and cancelled orders as revenue: v1 has no way to know the company's rule that revenue
  means completed orders only. v2 was better on 12 dev questions and worse on 1 (sign test p = 0.003),
  and better on 9 holdout questions and worse on none (p = 0.004).
- **Few-shot examples (v2 → v3) add little, and the gain is not statistically significant.** On dev,
  v3 was better on 2 questions and worse on 1 (p = 1.0); on holdout both scored 100%. v3 costs about
  16% more tokens per question than v2 (1,501 vs 1,295 on dev) and was slower on dev (8.9 s vs 6.1 s
  per question).
- **Holdout did not drop below dev**, so the prompts were not fitted to the dev questions. Holdout
  scores are higher partly because its questions were checked for ambiguity before the run and the
  dev questions were not (dev q20 turned out to have two readings, see below).
- **The self-correction loop was needed once in 360 answers** (holdout, v1, h14: a SQL error on the
  first try, fixed on the second). Strong models rarely write SQL that fails to run, so first-try and
  final accuracy are almost identical.
- With 20 questions per suite, intervals are wide: 20/20 is consistent with a true accuracy anywhere
  from about 84% to 100%. The numbers compare prompt versions on this task; they are not a general
  benchmark score.

Full reports: [`latest_report.md`](evals/results/latest_report.md) (the original run),
[`regrade_report.md`](evals/results/regrade_report.md) (re-grading, grader checks, intervals,
per-question table) and [`audit_evidence.md`](evals/results/audit_evidence.md) (every wrong answer
with its result next to the correct result).

### Test sets

| Suite | File | Questions | Purpose |
|---|---|---|---|
| `dev` | `evals/dataset.json` | 20 (5 easy, 8 medium, 7 hard) | Questions the prompts were developed against |
| `holdout` | `evals/holdout.json` | 20 (5 easy, 8 medium, 7 hard) | Written **after the prompts were frozen** and never used for tuning, so it measures how well the prompts generalise to questions they weren't built for |

Every question has a hand-written **gold SQL** query that follows the definitions in
`data/business_glossary.md`. The correct answer is whatever that query returns from the database, so
the answer key is computed, not judged. Held-out questions were checked for ambiguity: no ties in
"top N" questions, no answers of 0 that a broken query could hit by accident, and wording with only
one reasonable reading.

### Metric: execution accuracy

The model's SQL is run and its result rows are compared with the gold rows. This is the standard
metric of public text-to-SQL benchmarks such as Spider and BIRD. Column names and order may differ
and extra columns are allowed. Row order only counts for rankings and time series. No person or LLM
judges an answer.

Two versions of the number comparison exist, both in `src/querentcrew/evaluation.py`:

| Grader | Rule | Problem |
|---|---|---|
| `round1` (original, used during the run) | numbers equal after rounding to 1 decimal place | Far too coarse for small values such as rates: 0.046 vs 0.05 fails, 0.05 vs 0.14 passes |
| `tolerance` (corrected, current) | whole numbers must be equal; other numbers may differ by at most 0.01%, or by rounding (if one side was rounded, the other must round to it, with an allowance of at most 0.005) | — |

**Why the grader was changed after the run, and why that is not tuning.** The flaw was found while
auditing the results: in every v2 and v3 run, the model's return rate of 0.05 for a true value of
0.046 was marked wrong. Changing a grader after seeing results could be used to inflate them, so:

1. the original grader and its numbers are kept and shown first, unchanged;
2. the corrected grader was validated on the gold queries alone, without model answers (next
   section): it rejects every broken query and accepts every harmless rewrite, while the original
   grader fails both checks;
3. the correction moves verdicts both ways in principle (0.05 vs 0.14 now fails), and every changed
   verdict is listed with the actual result rows in `regrade_report.md`: 18 answers, all wrong →
   correct; on inspection 15 are clearly correct and 3 are the ambiguous q20 case described below;
4. both graders run on the same saved answers, so anyone can compare them with one command.

### How we know the grader is right

| Check | What it does | Result |
|---|---|---|
| Reproducibility | Re-grades the 360 saved answers with the original grader | Same verdict for **360/360** |
| Mutation testing of the grader | Breaks each gold query on purpose (wrong status filter, missing discount, SUM→AVG, wrong region, shifted dates, off-by-one LIMIT, …) and must reject it; makes harmless rewrites (columns reordered, extra column, rounding changed) and must accept them | Corrected: **189/189** broken rejected, **126/126** rewrites accepted. Original: 185/189 and 121/126 |
| Test-suite accuracy (Zhong et al., 2020) | Re-runs every answer marked correct, and its gold query, on **5 extra databases** generated with different random seeds, to catch queries that were right only by luck on one dataset | **0** verdicts changed |
| Manual review | Every answer marked wrong, every changed verdict and a random sample of 20 marked correct (106 of 360 answers), checked by hand | Grader agrees on **95 of 100** unambiguous answers; **0** false positives; 5 false negatives; 6 ambiguous |

The 5 false negatives are correct answers in a different format, which the grader cannot know are
equal: quarters as `1` instead of `Q1` (holdout v1 h09, 2 runs), `Q2 2025` instead of `Q2` (dev v1
q17, 1 run), and a return rate as a percentage instead of a fraction (dev v1 q13, 2 runs). The 6
ambiguous answers are dev **q20**, *"What is the average discount given in each product category?"*:
v2 and v3 averaged over completed orders only, the gold query over all orders, and the glossary does
not decide it. The grader was **not** changed to accept these cases, because that would be fitting
the grader to the answers. They are reported in the "manual review" column instead, with q20 shown
both ways. Labels and reasons are in [`evals/manual_review.json`](evals/manual_review.json).

The mutation tests run in CI (`tests/test_evaluation.py`), so a future change to the grader that
starts accepting wrong queries fails the build.

### Uncertainty

- **The questions, not the runs, are the sample.** Each question's score is its mean over the 3 runs
  and intervals use n = 20 questions, so repeated runs are not counted as extra evidence.
- **95% Wilson score intervals**, which stay sensible at 0% and 100% with small samples (the usual
  normal-approximation interval would give 100% ± 0 for 20/20).
- **Paired comparisons:** prompt versions are compared question by question with an exact sign test,
  as recommended for model evals that share the same questions.

| Suite | Comparison | Questions better | Worse | Same | Sign-test p |
|---|---|---|---|---|---|
| dev | v1 → v2 | 12 | 1 | 7 | 0.0034 |
| dev | v2 → v3 | 2 | 1 | 17 | 1.0000 |
| holdout | v1 → v2 | 9 | 0 | 11 | 0.0039 |
| holdout | v2 → v3 | 0 | 0 | 20 | 1.0000 |

### Safeguards against a biased result

- **Inputs were fixed before the results existed, and that is verifiable.** The prompts, glossary
  and both question sets used in the final run have the same content as commit `fb88019` (`git diff
  fb88019` shows no change to them),
  pushed to GitHub on **2026-11-19 20:34 UTC**. The final run started on 2026-11-20 and its report
  was produced on 2026-11-24. Push times are recorded by GitHub, not by the author's machine:
  `gh api repos/Himanshu-joshi986/QuerentCrew/activity`.
- **No test question appears in a prompt.** One v3 example (*"Total revenue by region in 2025"*)
  originally matched a dev question. It was replaced before the final run, and
  `test_no_test_question_leaks_into_prompt_examples` now fails if any example comes close to a test question.
- **Dev and holdout don't overlap.** `test_dev_and_holdout_do_not_overlap` checks this. It caught a
  held-out question that differed from a dev question by one word, and that question was replaced.
- **Inputs are fingerprinted.** Reports list SHA-256 hashes of the prompts, glossary, both test sets
  and the database. The database hash covers its data, not the file bytes, so a database regenerated
  with `scripts/seed_database.py` (fixed seed 42) gets the same fingerprint. Text files are hashed
  with line endings normalised, so Windows and Linux checkouts match. (The original report lists
  `sql_writer.yaml` as `e17a296d…`, the hash of the same file with Windows line endings; normalised it
  is `28c5916d…`. `regrade.py` accepts either.)
- **Every run is reported.** The numbers come from one run started with `--fresh` and resumed
  (without `--fresh`) when the provider's daily quota ran out. No run was chosen or discarded, and
  every saved answer is published in `evals/results/checkpoint.jsonl`.

### Reproduce it

```bash
python scripts/seed_database.py                  # same data for everyone (fixed seed)
python evals/run_eval.py --fingerprints          # compare with the hashes in the reports
python evals/regrade.py                          # re-grade the saved answers: free, no API key, ~1 minute
python evals/run_eval.py --repeats 3 --fresh     # optional: a new run (both suites, v1 v2 v3, 3 runs)
```

`regrade.py` needs no API key. It checks the input fingerprints, reproduces the original verdicts,
applies the corrected grader, the 5 extra databases and the manual review, and writes
`regrade_report.md` and `audit_evidence.md`.

A new run saves every answer as it goes (`evals/results/checkpoint.jsonl`). If the provider's daily
quota runs out (Groq's free tier allows 200k tokens/day for `gpt-oss-120b`, and one full run uses
about 130–150k), the run stops cleanly; run the same command **without** `--fresh` later and it
continues where it stopped. Saved progress is tied to the model, settings, grader and file
fingerprints, so a resumed run can't mix answers from different inputs or graders. A rerun with the
same model should land within the reported intervals; a different model gives different numbers,
because the harness measures model plus prompt together.

### Reading the reports

| File | Contents |
|---|---|
| `evals/results/latest_report.md` | The original run: accuracy per suite and prompt version (first-try, final, by difficulty, run range), errors, attempts, tokens, latency, every wrong answer with its SQL, fingerprints |
| `evals/results/regrade_report.md` | Reproducibility check, grader mutation tests, accuracy under each grader with 95% intervals, grader-vs-manual agreement, paired comparisons, per-question table, every changed verdict |
| `evals/results/audit_evidence.md` | For each wrong answer: the gold result next to the model's result; plus the random sample of correct answers |
| `evals/manual_review.json` | The manual labels that differ from the grader, with reasons |

With `LANGSMITH_TRACING=true`, every agent call appears in LangSmith as `querentcrew.<agent>`, so you
can inspect the exact prompts behind each score.

### What this evaluation does and does not show

- It **does** show, with checked grading, that the glossary and rules (v2) make the SQL agent far
  more accurate on this database, and that the gain holds on questions written after the prompts
  were frozen.
- It **does not** show that v3's few-shot examples help: the measured gain is within noise.
- It uses **one model, one small synthetic database and 40 questions**. It is a controlled
  comparison of prompt versions, not a general text-to-SQL benchmark score.
- It measures **SQL correctness only**, not the quality of the written report or the charts.
- The gold queries and the manual review were written by the same person who built the system. Both
  are published so they can be checked independently.

## 9. Tech stack

| Area | Tools |
|---|---|
| Agents and orchestration | LangChain 1.x (`init_chat_model`, structured output, retry/fallback), LangGraph 1.x (`StateGraph`, `interrupt`, `Command`, checkpointer) |
| LLMs | Groq (`openai/gpt-oss-120b`, free tier; used for the evaluation), Gemini, OpenAI, Anthropic, Ollama (local), Bedrock (optional) |
| Observability | LangSmith (optional), built-in `UsageTracker` |
| Data | SQLite, Pandas, NumPy |
| Charts | Altair / Vega-Lite (interactive, in the UI), Matplotlib (PNG for API and CLI) |
| Validation | Pydantic v2 |
| Interfaces | Streamlit (custom dark theme, Material Symbols icons), FastAPI + Uvicorn |
| Quality | pytest (fake LLM), GitHub Actions, ruff |
| Deployment | Docker, Docker Compose, AWS EC2 / SSM Parameter Store / S3 / CloudWatch |

## 10. Project structure

```
querentcrew/
├── app/
│   ├── streamlit_app.py          # chat UI: example cards, interactive charts, data/SQL/plan/trace tabs, approval card, CSV upload
│   └── style.css                 # UI theme (dark glass design, responsive layout)
├── src/querentcrew/
│   ├── config.py                 # Settings from env / .env
│   ├── schemas.py                # Pydantic models for structured LLM output
│   ├── state.py                  # AgentState shared by the graph
│   ├── llm.py                    # LLMClient interface, LangChainLLM, UsageTracker
│   ├── nodes.py                  # the agents + routing functions
│   ├── graph.py                  # LangGraph wiring
│   ├── service.py                # QuerentCrew facade (ask / resume / history / import_csv)
│   ├── database.py               # read-only execution, schema introspection, CSV import
│   ├── sql_safety.py             # SQLGuard + PII masking
│   ├── analysis.py               # Pandas/NumPy profiler
│   ├── charts.py                 # chart spec validation + heuristic fallback, Matplotlib PNG renderer
│   ├── evaluation.py             # execution-accuracy evaluator, graders, intervals, fingerprints
│   ├── grader_validation.py      # mutation tests that check the grader itself
│   ├── api.py                    # FastAPI app
│   ├── __main__.py               # CLI: python -m querentcrew
│   └── prompts/                  # registry.py + versioned YAML prompts
├── evals/
│   ├── dataset.json              # dev suite: 20 gold questions
│   ├── holdout.json              # held-out suite: 20 gold questions written after the prompts were frozen
│   ├── run_eval.py               # compare prompt versions (repeats, resume, fingerprints)
│   ├── regrade.py                # re-grade saved answers for free: grader checks, extra databases, intervals
│   ├── manual_review.json        # manual labels that differ from the grader, with reasons
│   └── results/                  # published reports and every saved answer (checkpoint.jsonl)
├── data/business_glossary.md     # metric definitions used in prompts
├── scripts/seed_database.py      # reproducible sample database with a planted story
├── tests/                        # 83 tests, fake LLM, no API key needed
├── docs/DEPLOY_AWS.md            # step-by-step AWS deployment
├── Dockerfile, docker-compose.yml, Makefile, pyproject.toml, requirements.txt
└── .github/workflows/ci.yml      # lint + tests on 3.11/3.12 + Docker build
```

## 11. Getting started

**Prerequisites:** Python 3.11+ and one LLM option. The quickest free one is a **Groq API key**
from <https://console.groq.com>. Alternatives: a Gemini key from <https://aistudio.google.com>, or
[Ollama](https://ollama.com) for a fully local model.

```bash
git clone https://github.com/Himanshu-joshi986/querentcrew.git
cd querentcrew

python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
pip install -e .

cp .env.example .env            # Windows: copy .env.example .env
# edit .env -> set LLM_MODEL and the matching API key

python scripts/seed_database.py # creates data/sales.db (optional: the app also builds it on first start)
pytest                          # optional: all tests pass without an API key
streamlit run app/streamlit_app.py
```

Open <http://localhost:8501> and click one of the example question cards, or type your own question.

## 12. Usage

### Streamlit UI
- **Welcome screen:** six example question cards (root cause, trend, ranking, segments, channels, governed data). Click one or type your own question.
- **Live progress:** while the agents work, each step (planner, SQL writer, guard, executor...) appears as it happens.
- **Every answer** is a Markdown report followed by a stats strip (queries OK, rows, LLM calls, tokens, time) and tabs:
  - **Chart:** interactive chart (hover for exact values, highlighted series and bars, donut with total)
  - **Data:** result table per step, numbers with thousands separators, **CSV download**
  - **SQL:** the exact query per step, number of attempts and run time
  - **Plan:** the sub-questions and the analyst's caveats
  - **Agent trace:** timeline of every agent step, with raw details on demand
- **Approval card** with **Approve and run / Reject** and an optional comment when a query needs review
- **Sidebar:** model connection status, new conversation, **export chat as Markdown**, session totals,
  data explorer (click a table to preview 50 rows and its columns), **CSV import** and governance settings
- Responsive layout: works on laptop and phone screens (the sidebar collapses on small screens)

### CLI
```bash
python -m querentcrew                                    # interactive chat
python -m querentcrew "Which 5 products earned the most revenue?"
```
Charts are saved to `outputs/`.

### REST API
```bash
uvicorn querentcrew.api:app --reload --port 8000         # docs at http://localhost:8000/docs
```

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/schema` | Tables, columns, row counts |
| POST | `/ask` | `{"question": "...", "thread_id": "optional"}` |
| POST | `/approve` | `{"thread_id": "...", "approved": true, "comment": ""}` |
| GET | `/history/{thread_id}` | Conversation memory |
| POST | `/upload-csv` | Multipart CSV → new table |

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "Show monthly revenue for 2025 by region"}'
```

The response contains `status` (`completed` / `awaiting_approval`), `answer`, `plan`, `results`
(SQL + rows), `insights`, `chart.image_b64`, `trace` and `usage`.

### Using your own data
Upload a CSV in the UI (sidebar → **Import a CSV**) or with `POST /upload-csv`, or point
`DATABASE_PATH` at your own SQLite file. On import, Pandas cleans the headers (for example
`Annual Salary (USD)` becomes `annual_salary__usd`) and infers the column types. The table can then be
queried right away, because the agents re-read the schema.

Checked with a non-sales dataset (200 employees: department, hire date as `dd/mm/yyyy`, salary,
phone). Questions such as *"average salary by department"*, *"employees hired each year"* and
*"employees hired between March 2021 and June 2022"* returned correct SQL, results and charts.
*"Phone numbers of the top 3 engineers"* paused for approval and returned masked numbers.

For the best accuracy on your own data:

1. **Rewrite `data/business_glossary.md`** with your metric definitions. The bundled glossary and
   the SQL writer's few-shot examples are written for the sales sample, so this is the single most
   useful change.
2. **List your personal-data columns in `PII_COLUMNS`.** Columns are matched by **name**, so a
   column called `mobile`, `address` or `ssn` is not protected until you add it.
3. **Prefer ISO dates (`YYYY-MM-DD`).** Dates are stored as they appear in the CSV. The model
   usually handles other formats, but an ambiguous date such as `03/04/2024` can be read the wrong way.
4. Importing a file with the same name as an existing table **replaces** that table.

### Where your data lives (local storage and privacy)
- **Storage is local.** All data lives in a single SQLite file (`DATABASE_PATH`, default
  `data/sales.db`). Uploaded CSVs become new tables in that same file and stay there after a restart.
  Nothing is downloaded from the internet, and the database is opened **read-only** for queries.
- **What goes to the LLM provider.** With a hosted model (Groq, Gemini, OpenAI...), each question
  sends the provider the table and column names, **2 sample rows per table**, and a preview of each
  query result so the analyst and reporter can describe it. PII columns are masked **before** results
  reach the model. The full database is never uploaded.
- **Fully offline option.** Set `LLM_MODEL=ollama:<model>` to run a local model with
  [Ollama](https://ollama.com), so no data leaves your machine.

## 13. Configuration

All settings live in `.env` (see `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `LLM_MODEL` | `groq:openai/gpt-oss-120b` | `provider:model` for `init_chat_model` |
| `LLM_FALLBACK_MODEL` | – | Used automatically if the primary model fails |
| `LLM_TEMPERATURE` | `0` | Deterministic output for SQL |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` | `false` | Turn on LangSmith tracing |
| `DATABASE_PATH` | `data/sales.db` | SQLite file |
| `MAX_RESULT_ROWS` | `500` | Row cap per query |
| `QUERY_TIMEOUT_SECONDS` | `10` | Abort slow queries |
| `MAX_SQL_RETRIES` | `2` | Self-correction attempts after the first try |
| `MAX_SUB_QUESTIONS` | `3` | Planner limit |
| `PROMPT_VERSION_SQL` | `v3` | SQL writer prompt version |
| `PII_COLUMNS` | `email,phone` | Columns that need approval and are masked |
| `REQUIRE_APPROVAL_FOR_PII` | `true` | Approval gate for PII |
| `REQUIRE_APPROVAL_FOR_ALL` | `false` | Approval gate for every query |

## 14. Testing

```bash
pytest -v
```

The LLM sits behind the `LLMClient` interface, so tests inject a scripted `FakeLLM`
(`tests/fakes.py`) and run the **real** graph, database, guard and analysis code with no API key or
network:

| File | Covers |
|---|---|
| `test_sql_safety.py` | blocked statements, string literals, CTEs, LIMIT injection, PII detection, masking |
| `test_database.py` | read-only enforcement even if the guard is bypassed, row cap, errors, CSV import |
| `test_analysis.py` | period-over-period, shares, wide comparisons, type cleaning, chart fallback |
| `test_prompts.py` | every prompt version renders with no missing variables |
| `test_nodes.py` | retry with error feedback, guard rewrite, retries exhausted, approval/rejection, masking, chart fallback when the model declines |
| `test_graph.py` | full pipeline, self-correction, interrupt/resume, rejection path, multi-turn memory |
| `test_evaluation.py` | result matching under both graders, number tolerance, grader mutation tests (every broken gold query rejected, every harmless rewrite accepted), gold SQL validity (dev + holdout), no prompt/test leakage, no dev/holdout overlap, line-ending-independent fingerprints, Wilson interval and sign test, rate-limit retry, daily-quota stop, report writing |
| `test_api.py` | REST endpoints including the ask → approve flow |
| `test_llm_errors.py` | provider error classification, fail-fast with a clear message when the model is unreachable or drops mid-run, step-by-step progress streaming |

CI (`.github/workflows/ci.yml`) runs ruff and the test suite on Python 3.11 and 3.12, then builds the Docker image.

## 15. Deployment (Docker, AWS, Streamlit Cloud)

```bash
docker compose up --build       # UI on :8501, API on :8000
```

Full AWS guide: **[docs/DEPLOY_AWS.md](docs/DEPLOY_AWS.md)**. It covers EC2 (free-tier instance) with
Docker Compose, the API key in **SSM Parameter Store**, an IAM instance role, a security group,
**S3** backups, **CloudWatch** logs, optional **Bedrock**, a cost budget and clean-up.

### Free hosting on Streamlit Community Cloud

1. Push the repository to GitHub, then go to <https://share.streamlit.io> → **Create app**.
2. Choose the repository and set the main file to `app/streamlit_app.py`.
3. Under **Advanced settings → Secrets**, add your model settings (they become environment variables):
   ```toml
   LLM_MODEL = "groq:openai/gpt-oss-120b"
   GROQ_API_KEY = "your-key"
   ```
4. Deploy. The sample database is created automatically on first start.

On the free tier, the app sleeps after a period with no visitors, and uploaded CSVs last only until
the next restart. Everyone who opens a public app uses your API quota, so use a free-tier key.

## 16. Design decisions

- **Explicit `StateGraph` instead of one ReAct agent:** a data workflow has known stages. Explicit
  nodes make routing, retries and approval points predictable, testable and visible in traces. The
  LLM decides *content* (plan, SQL, findings), and the graph decides *control flow*.
- **The LLM never does arithmetic:** `DataProfiler` computes the changes and shares the analyst cites,
  which removes the most common source of wrong numbers in LLM reports.
- **Rule-based guard, not an LLM judge, for safety:** safety checks must be deterministic and unit-testable.
- **Dependency injection:** `AgentNodes` receives its LLM, database, guard, prompts and interrupt
  function, so the same graph runs in production and in tests.
- **Plain JSON-friendly state:** state can be checkpointed and sent over the API unchanged.
- **Prompts as versioned data with an eval harness:** prompt changes are judged by measured accuracy, not by feel.

## 17. Limitations and roadmap

- SQLite only. Next: a Postgres/MySQL connector using a read-only DB user.
- In-memory checkpointer, so memory resets on restart. Next: `langgraph-checkpoint-postgres` or SQLite saver.
- The guard's table/column parsing is regex-based. Next: `sqlglot` AST parsing.
- Single-tenant demo, no auth. Next: Cognito/OIDC on the AWS deployment.
- PII detection is by column name (`PII_COLUMNS`). Next: value-based detection (email/phone patterns).
- CSV dates are stored as given. Next: normalise date columns to ISO `YYYY-MM-DD` on import.
- The prompts' few-shot examples are sales-specific. Next: generate examples from the glossary per dataset.
- The evaluation measures SQL accuracy only, on one model and 40 questions. Next: more questions per suite (narrower intervals), a second model, and LLM-as-judge scoring of report quality validated against human ratings.
- Next features: a schema-RAG retriever for databases with hundreds of tables, scheduled report emails.

---

**Author:** Himanshu Joshi · [GitHub](https://github.com/Himanshu-joshi986) · [LinkedIn](https://linkedin.com/in/himanshu-joshi-735066292/)
