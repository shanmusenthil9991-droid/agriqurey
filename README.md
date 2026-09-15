# CropCare AI — Crop Disease Question Classifier

Classifies a farmer's crop-disease **question** into the type of information the
farmer is asking for, using a large language model with few-shot prompting.

> **This system classifies questions. It does not diagnose crop diseases.**

---

## 1. Project Title

**CropCare AI — Prompt-Based Classification of Farmer Crop-Disease Questions**

A web application and evaluation harness that takes a free-text farmer question
and predicts which of four information-need categories it belongs to, together
with a self-reported confidence score and a one-sentence rationale.

---

## 2. Problem Statement

Farmers ask about crop disease in many different ways. One describes a brown
patch on a leaf; another wants to stop a disease before planting; a third is
already treating an outbreak and needs next steps; a fourth simply wants to know
what a pathogen is.

These are fundamentally different *information needs*, but they arrive as one
undifferentiated stream of free text. An advisory service that cannot tell them
apart sends everyone the same generic answer — a symptom guide to someone who
needs a treatment schedule, or a fungicide table to someone who only wanted a
definition.

The problem this project solves is **routing**: given a question in plain
language, determine what *kind* of answer the farmer is looking for, so the
question can be directed to the right resource or expert.

The problem this project explicitly does **not** solve is diagnosis. Naming the
disease affecting a farmer's crop requires agronomic expertise, local context,
and usually physical inspection. Getting that wrong causes real economic harm,
so the system never attempts it.

---

## 3. Objectives

1. Build a labelled dataset of realistic farmer crop-disease questions across
   four information-need categories.
2. Split that dataset into development, validation, and held-out test sets with
   strict separation between them.
3. Design a prompt that makes an LLM classify *information need* without
   drifting into diagnosis.
4. Use few-shot prompting, with examples drawn only from the development split.
5. Serve classification through a Flask API so that no API key or LLM call ever
   reaches the browser.
6. Provide a farmer-facing web interface for asking a question and viewing the
   result.
7. Evaluate the classifier on the held-out test set with standard metrics.
8. Compare few-shot prompting against a zero-shot baseline under identical
   conditions.
9. Analyse errors to identify which categories are confused with which.
10. Report every number honestly — measured, never fabricated.

---

## 4. System Architecture

Two independent pipelines share the same classification core.

**Runtime pipeline** (a farmer asking a question):

```
Browser (frontend/ask.html)
    |  POST /api/classify  {question, crop}
    v
Flask API (backend/routes/classify.py)        <- validates the request
    v
Prompt Builder (backend/services/prompt_builder.py)
    v
Few-Shot Examples (data/development.csv only)
    v
LLM Provider Adapter (backend/services/llm_providers.py)
    v
LLM (Anthropic / OpenAI-compatible)            <- server-side only
    v
Structured Classification (backend/services/llm_classifier.py)
    |  parse JSON -> validate category -> validate confidence
    v
JSON response {category, confidence, reason}
    v
Result Page (frontend/result.html)
```

**Evaluation pipeline** (measuring quality):

```
Dataset (data/crop_questions.csv, 300 rows)
    v
Development / Validation / Test Split (backend/data_preprocessor.py)
    v
LLM Evaluation (backend/evaluation/evaluate_classifier.py)   <- test.csv only
    v
Metrics (accuracy, macro precision/recall/F1, confusion matrix)
    v
evaluation/results.json + confusion_matrix.json + predictions.csv
    v
GET /api/evaluation  ->  Evaluation Dashboard (frontend/admin.html)
```

The Flask app serves **both** the API and the frontend pages on one origin, so
the browser's relative `/api/...` calls resolve without CORS configuration or a
hardcoded backend URL.

### Project structure

```
crop-disease-classifier/
├── run.py                      # Entry point; prints a readiness banner
├── requirements.txt
├── .env.example                # Copy to .env — never commit .env
├── backend/
│   ├── app.py                  # Application factory, error handlers
│   ├── config/settings.py      # Env-var configuration (no secrets in code)
│   ├── data_loader.py          # Load + validate the dataset
│   ├── data_preprocessor.py    # Clean + stratified split
│   ├── prompts/
│   │   └── classification_prompt.txt
│   ├── routes/
│   │   ├── health.py           # GET  /api/health
│   │   ├── classify.py         # POST /api/classify
│   │   ├── evaluation.py       # GET  /api/evaluation
│   │   └── frontend.py         # Serves the HTML pages
│   ├── services/
│   │   ├── prompt_builder.py   # Prompt assembly + few-shot selection
│   │   ├── llm_providers.py    # Vendor adapters (all vendor code lives here)
│   │   └── llm_classifier.py   # Call, parse, validate -> structured result
│   └── evaluation/
│       ├── evaluate_classifier.py
│       ├── compare_methods.py
│       └── error_analysis.py
├── data/
│   ├── crop_questions.csv      # 300 labelled questions
│   ├── development.csv         # 210 rows (70%)
│   ├── validation.csv          #  45 rows (15%)
│   └── test.csv                #  45 rows (15%) — held out
├── frontend/                   # index / ask / result / admin
├── static/                     # css + js
├── evaluation/                 # Generated artifacts (gitignored)
└── tests/                      # 251 tests
```

---

## 5. Technologies

| Layer | Technology | Purpose |
|---|---|---|
| Backend | Flask 3.0 | API and page serving |
| Config | python-dotenv | Environment-variable loading |
| CORS | flask-cors | Optional cross-origin support |
| Data | pandas | Dataset loading and manipulation |
| Split / Metrics | scikit-learn | Stratified split, accuracy, P/R/F1, confusion matrix |
| HTTP client | requests | Server-side LLM API calls |
| Testing | pytest | 251 automated tests |
| Frontend | HTML5, Bootstrap 5.3, vanilla JS | Responsive UI, no build step |
| Charts | Chart.js 4.4 | Category distribution chart |

No machine-learning training framework is used. Classification is performed
entirely through prompting a hosted LLM.

---

## 6. Dataset

`data/crop_questions.csv` — **300 labelled farmer questions**.

| Column | Type | Description |
|---|---|---|
| `id` | int | Unique row identifier |
| `question` | str | The farmer's question in plain language |
| `crop` | str | Crop the question concerns |
| `category` | str | Ground-truth label (one of the four categories) |

Crops covered: Cotton, Maize, Potato, Rice, Tomato, Wheat.

The dataset is **perfectly balanced** — 75 questions per category. Questions are
written in the varied, informal register farmers actually use, including
incomplete sentences and regional phrasing.

`backend/data_loader.py` validates on every load and will refuse a malformed
file:

- **Hard failure** if a required column is missing.
- **Hard failure** if any `category` value falls outside the four allowed labels.
- **Reported and optionally dropped**: rows with missing values, and duplicate
  questions (compared case-insensitively and whitespace-trimmed).

---

## 7. Dataset Split

`backend/data_preprocessor.py` produces a **stratified** three-way split using
`random_state=42`, so the split is fully reproducible.

| Split | Rows | Share | Purpose |
|---|---|---|---|
| Development | 210 | 70% | Prompt design and few-shot example source |
| Validation | 45 | 15% | Tuning prompt choices without touching test |
| Test | 45 | 15% | Final evaluation only |

Stratification is on `category`, so all four categories appear proportionally in
every split:

| Category | Development | Validation | Test |
|---|---|---|---|
| Symptoms | 52 | 12 | 11 |
| Prevention | 52 | 11 | 12 |
| Management | 53 | 11 | 11 |
| General Information | 53 | 11 | 11 |

> **The test set is never used for few-shot examples, prompt design, or any
> other form of development.** This is enforced in code, not left to
> convention — see §10.

---

## 8. The Four Categories

| Category | The farmer is asking... | Example |
|---|---|---|
| **Symptoms** | What does the disease look like? | "Maize plant showing white powdery coating, what disease is this?" |
| **Prevention** | How do I stop it before it happens? | "How to save Wheat from disease before season" |
| **Management** | How do I deal with it now that it's here? | "Which spray should I use for my infected tomato crop?" |
| **General Information** | What is this, generally? | "What causes bacterial blight in rice?" |

A fifth label, **`Unable to Classify`**, exists but is **not** one of the four
measured categories. The model may return it for questions that are unrelated to
crop disease or unintelligible. It never appears as a ground-truth label in the
dataset and is deliberately excluded from the primary metric label set; how
often it is predicted is tracked separately as a diagnostic.

The same four labels are used consistently across the dataset, the prompt, the
classifier's validation logic, the evaluation scripts, and the frontend.

---

## 9. Prompt-Based Classification

The prompt template lives in `backend/prompts/classification_prompt.txt` — kept
as a separate file so it can be versioned and reviewed independently of code.

It contains six parts:

1. **Role** — an agricultural information-intent classifier, explicitly *not* a
   plant pathologist.
2. **Category definitions** — what each of the four labels means.
3. **Classification rules** — including the instruction to classify information
   need only, never to diagnose or guess the disease.
4. **Few-shot examples** — injected at runtime.
5. **The farmer's question**, with crop context.
6. **Required output format** — strict JSON only.

The model must return exactly:

```json
{
  "category": "Symptoms",
  "confidence": 0.91,
  "reason": "The question asks about visible signs of crop disease."
}
```

Temperature is fixed at **0.0** — classification is a deterministic task, not a
creative one.

### On the `confidence` field

Confidence is the model's **self-reported** estimate. It is a soft signal for UI
hinting only. It is **not** a calibrated probability and **not** a measure of
accuracy. Real accuracy comes only from evaluation against the held-out test
set. The codebase documents this at every point where confidence is handled.

---

## 10. Few-Shot Prompting

`select_few_shot_examples()` in `backend/services/prompt_builder.py` selects
**2 examples per category (8 total)** by default:

1. Reads `data/development.csv` and keeps only the four allowed categories.
2. Samples per category with a fixed seed, so prompts are reproducible.
3. Enforces **balance** — every category contributes the same count, so the
   prompt never over-represents one label.
4. **Shuffles** the final list, so category order does not become a positional
   cue the model can exploit instead of meaning.

Examples are cached per process: the seeded selection is identical on every call
anyway, so caching also keeps prompts stable across requests.

### Leakage prevention

Test-set separation is **structural, not conventional**. `_guard_not_test_split()`
raises `DataLeakageError` if the few-shot source path resolves to `test.csv` or
`validation.csv`, regardless of how the argument was supplied:

```python
FORBIDDEN_SOURCE_FILENAMES = {"test.csv", "validation.csv"}
```

Using the test set to pick examples would leak it into prompt design and
silently inflate the final score. A stray argument cannot bypass this guard.

---

## 11. LLM Integration

All vendor-specific knowledge is isolated in
`backend/services/llm_providers.py` — endpoint shape, auth header names, request
body format, and where generated text sits in the response.

| Provider | Value | Notes |
|---|---|---|
| Anthropic | `anthropic` | Messages API |
| OpenAI-compatible | `openai` | OpenAI, OpenRouter, Groq, Together, Ollama, vLLM |
| Offline stub | `echo` | No network call, no key; **does not classify** |

Swapping provider means changing `LLM_PROVIDER` in `.env`. No change to the
classifier, prompt builder, routes, or tests.

### Response handling

`backend/services/llm_classifier.py` is defensive because model output drifts:

- **Extraction** strips markdown code fences and leading/trailing commentary,
  then scans for the first balanced `{...}` block — respecting strings and
  escapes so braces inside `reason` don't break matching.
- **Category validation** normalises case and accepts a small, conservative list
  of unambiguous synonyms. Anything unrecognised is **rejected, not guessed** —
  silently coercing an unknown label would fabricate a result.
- **Confidence validation** coerces to a float in `[0, 1]`, clamping overshoot
  and converting obvious percentages. Non-numeric values are rejected.
- **Retries** apply only to *transient* failures (timeouts, HTTP 429, HTTP 5xx),
  with exponential backoff. Permanent errors like 401 and 400 are never retried.

`classify_question()` **never raises**. Every failure becomes a structured error
dict, so a route can render it directly and malformed model output cannot crash
the application.

Critically, `status` distinguishes a genuine `Unable to Classify` *judgement*
(`status: "ok"`) from a *system failure* (`status: "error"`). Collapsing those
two would turn outages into confident-looking classifications and corrupt any
evaluation run.

### Security

- The API key is read from an environment variable and **never** hardcoded,
  logged, returned in a response, or included in an exception message.
- **The browser never calls an LLM provider.** The only two `fetch()` calls in
  the frontend target same-origin `/api/classify` and `/api/evaluation`.
- Provider error messages include status codes and short response snippets only
   — never headers or credentials.
- Request bodies are capped at 64 KB (`MAX_CONTENT_LENGTH`), with a dedicated
  JSON `413` handler, so an oversized payload cannot be fully buffered and
  parsed into memory before the application-level length check ever runs.

---

## 12. API

The Flask app serves the API and the frontend on the same origin
(default `http://localhost:5000`).

### `GET /api/health`

Liveness **and** readiness. Always returns 200 while the process is up.

```json
{
  "status": "ok",
  "service": "crop-disease-classifier-backend",
  "message": "Flask server is running.",
  "llm_provider": "anthropic",
  "llm_configured": true,
  "llm_is_offline_stub": false,
  "few_shot_examples_available": true
}
```

`status` is `"degraded"` when the server is running but no LLM key is
configured. Only booleans and names are reported — never the key, any prefix of
it, or its length.

### `POST /api/classify`

Request:

```json
{ "question": "Why are my tomato leaves turning yellow?", "crop": "Tomato" }
```

`crop` is optional. Unknown fields are rejected rather than silently ignored.

Success (`200`):

```json
{
  "category": "Symptoms",
  "confidence": 0.93,
  "reason": "The question asks about visible signs of crop disease."
}
```

| Status | Meaning |
|---|---|
| `200` | Classified successfully |
| `400` | Malformed request (missing, blank, wrong-typed, or unexpected fields) |
| `502` | LLM returned unparseable output or the upstream call failed |
| `503` | Server misconfigured (e.g. no API key) |

No stack trace, exception text, internal path, or credential ever appears in a
response body.

### `GET /api/evaluation`

Read-only. Returns evaluation artifacts already on disk. It **does not** run the
classifier and **does not** compute or invent any number.

If the evaluation has never been run it returns `200` with:

```json
{ "available": false, "message": "No evaluation results available yet. ..." }
```

An empty report is an expected early state, not a server error. The dashboard
renders that message rather than fabricating placeholder charts.

### Frontend routes

`/`, `/index.html`, `/ask.html`, `/result.html`, `/admin.html` — served from an
explicit allow-list, so path traversal is structurally impossible.

---

## 13. Evaluation

`backend/evaluation/evaluate_classifier.py` runs the **held-out test set** through
the exact same production pipeline the API uses.

```bash
python -m backend.evaluation.evaluate_classifier
```

It reads only `data/test.csv`; it never opens `development.csv`, so the test set
cannot leak into prompts.

**The offline `echo` provider is hard-refused.** It returns a fixed canned
response for every question, which would produce numbers that look real but
measure nothing. This is a hard stop, not a warning.

### Handling of non-predictions

- **System failures** (`status: "error"` — network, provider, or config errors
  where no judgement was ever made) are **excluded** from the scored metrics and
  reported separately. Scoring them as "wrong" would conflate infrastructure
  reliability with classification quality; dropping them silently would
  misrepresent the sample size. The script does neither.
- **`Unable to Classify` predictions** on genuine four-class questions are left
  in `y_pred`. They cannot match any true label, so they correctly reduce that
  class's recall, and the count is reported under `diagnostics`.

### Outputs

| File | Contents |
|---|---|
| `evaluation/results.json` | Metrics, classification report, diagnostics |
| `evaluation/confusion_matrix.json` | Labels and 4×4 matrix |
| `evaluation/predictions.csv` | Per-question prediction, correctness, status |

> **No number in the saved output is hardcoded.** Every value comes from
> actually calling the classifier on `data/test.csv` during that run.

---

## 14. Metrics

Computed with scikit-learn, restricted to the four primary categories via the
`labels=` parameter.

| Metric | Definition |
|---|---|
| **Accuracy** | Proportion of scored test questions classified correctly |
| **Precision (macro)** | Per-category precision, averaged equally across the four |
| **Recall (macro)** | Per-category recall, averaged equally across the four |
| **F1 (macro)** | Harmonic mean of precision and recall, macro-averaged |
| **Confusion matrix** | `matrix[i][j]` = true category `i` predicted as `j` |

Macro averaging weights each category equally regardless of support, which suits
a balanced four-class problem and prevents a single large category from
dominating the headline figure.

Also reported as **diagnostics** (not part of the primary metrics):
`Unable to Classify` prediction count, system-failure count, and the set of
error codes encountered.

Results are viewable at `/admin.html`, which renders KPI cards, a per-category
performance table, a category distribution chart, the confusion matrix, and the
baseline comparison — all read from `GET /api/evaluation`. Every figure on that
page comes from the fetched JSON; the dashboard script hardcodes no metric.

---

## 15. Baseline Comparison

`backend/evaluation/compare_methods.py` answers a single research question: does
few-shot prompting beat zero-shot on the same held-out test set?

```bash
python -m backend.evaluation.compare_methods
```

| | Experiment A (baseline) | Experiment B (proposed) |
|---|---|---|
| Few-shot examples | None (empty list) | 8, from `development.csv` |
| Prompt template | Identical | Identical |
| Category rules | Identical | Identical |
| Validation logic | Identical | Identical |
| Provider and model | Identical | Identical |
| Test set | `data/test.csv` | `data/test.csv` |

The **only** variable that changes is the presence of few-shot examples, which
isolates their effect from every other factor.

The script **does not assert that few-shot prompting is better.** It computes
both conditions from actual classifier runs, reports the measured difference,
and describes only that difference. `determine_conclusion()` has explicit
branches for improvement, degradation, no change, and mixed results. If the
numbers show no improvement, that is what gets printed and saved to
`evaluation/method_comparison.json`.

---

## 16. Error Analysis

```bash
python -m backend.evaluation.error_analysis
```

Reads `evaluation/predictions.csv` and isolates every misclassified question.

**Outputs:**

- `evaluation/errors.csv` — `id`, `question`, `crop`, `actual_category`,
  `predicted_category`
- `evaluation/error_analysis_summary.json` — totals, accuracy on scored rows,
  and confusion-pair counts

**Directional confusions** (`Actual -> Predicted`) are counted and ranked by
frequency.

**Three pairs are tracked by name** regardless of frequency, because they are the
conceptually hardest boundaries:

| Tracked pair | Why it's hard |
|---|---|
| Symptoms vs General Information | "What causes these spots?" sits between describing signs and explaining aetiology |
| Prevention vs Management | Both concern action; they differ only on whether disease is already present |
| Management vs General Information | Treatment questions often carry background framing |

These are counted **undirected** — a Symptoms question predicted as General
Information and the reverse both count toward the same total — while the
directional table above preserves each direction separately.

The script **reports what happened; it does not invent explanations for why**.
Interpretation belongs in the project write-up, not in generated output. It also
never modifies ground-truth labels, and excludes system failures for the same
reason the metrics do.

---

## 17. Installation

**Requirements:** Python 3.9+

```bash
# 1. Get the project
cd crop-disease-classifier

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your environment file
cp .env.example .env            # Windows: copy .env.example .env
```

`.env.example` ships with `LLM_PROVIDER=echo`, so the app runs immediately with
**no API key and no network calls**. The echo stub returns a fixed placeholder
and does not classify — see §18 to enable real classification.

---

## 18. Environment Variables

All configuration lives in `.env`, which is **gitignored and must never be
committed**. No secret is ever hardcoded in source.

| Variable | Default | Purpose |
|---|---|---|
| `FLASK_ENV` | `development` | Flask environment name |
| `FLASK_DEBUG` | `True` | Debug mode |
| `FLASK_RUN_HOST` | `0.0.0.0` | Bind host |
| `FLASK_RUN_PORT` | `5000` | Bind port |
| `LLM_PROVIDER` | `anthropic` | `anthropic`, `openai`, or `echo` |
| `LLM_API_BASE_URL` | *(provider default)* | Override the API endpoint |
| `LLM_MODEL_NAME` | *(empty)* | Model identifier |
| `LLM_API_KEY` | *(empty)* | **Secret.** Never commit this. |
| `LLM_REQUEST_TIMEOUT` | `30` | Per-request timeout, seconds |
| `LLM_MAX_RETRIES` | `1` | Extra attempts for transient failures only |
| `DATASET_PATH` | `data/crop_questions.csv` | Dataset location (resolved from project root) |

To enable real classification:

```bash
LLM_PROVIDER=anthropic
LLM_MODEL_NAME=<your-model-id>
LLM_API_KEY=<your-key>
```

A real key is **required** for evaluation — the `echo` stub is refused by both
evaluation scripts.

### Using a free API key (Groq)

If you don't have an Anthropic or OpenAI key, [Groq](https://console.groq.com)
offers a genuinely free, no-credit-card developer tier (rate-limited, not
time-limited) and speaks the same `/chat/completions` schema as OpenAI, so it
runs through the existing `openai` provider adapter with no code changes:

1. Sign up at https://console.groq.com and create a key at
   https://console.groq.com/keys.
2. Set the following in your `.env` (model names change over time — check
   https://console.groq.com/docs/models for the current list):

   ```bash
   LLM_PROVIDER=openai
   LLM_API_BASE_URL=https://api.groq.com/openai/v1/chat/completions
   LLM_MODEL_NAME=llama-3.1-8b-instant
   LLM_API_KEY=<your-groq-key>
   ```
3. Restart the app (`python run.py`). The startup banner will confirm the
   provider is configured.

Other OpenAI-compatible providers with free tiers (e.g. OpenRouter) work the
same way — just swap `LLM_API_BASE_URL` and `LLM_MODEL_NAME`.

---

## 19. Running the Application

```bash
python run.py
```

Then open **http://localhost:5000**.

Startup prints a readiness banner showing the active provider and whether a key
is configured. It prints the provider *name* and a *boolean* only — never the
key, any prefix of it, or its length.

| Page | URL | Purpose |
|---|---|---|
| Home | `/` | Overview and the four categories |
| Ask | `/ask.html` | Submit a question |
| Result | `/result.html` | View the classification |
| History | `/history.html` | Browse all past searches |
| Dashboard | `/admin.html` | Evaluation metrics |

The API and the frontend are served from the **same origin**, so the browser's
relative `/api/...` calls resolve with no CORS setup and no hardcoded backend
URL. Opening the HTML files directly from disk will **not** work — `file://`
cannot resolve `/api/classify`.

---

## 20. Running Evaluation

Requires a real LLM provider configured in `.env`.

```bash
# 1. (Re)generate the dataset splits — deterministic, seed 42
python backend/data_preprocessor.py

# 2. Evaluate on the held-out test set
python -m backend.evaluation.evaluate_classifier

# 3. Compare few-shot against the zero-shot baseline
python -m backend.evaluation.compare_methods

# 4. Analyse the errors (requires step 2 first)
python -m backend.evaluation.error_analysis

# 5. View the results
python run.py    # then open http://localhost:5000/admin.html
```

Steps 2–4 make real API calls and will incur usage costs.

---

## 21. Running Tests

```bash
pytest                  # all 251 tests
pytest -v               # verbose
pytest tests/test_evaluation_metrics.py    # one module
```

Tests require **no API key and make no network calls** — providers are injected
or stubbed throughout.

| Module | Covers |
|---|---|
| `test_data_loader.py` | Dataset loading, column/category validation, duplicates |
| `test_data_preprocessor.py` | Cleaning, stratified split, ratios, reproducibility |
| `test_prompt_builder.py` | Prompt assembly, few-shot selection, balance, leakage guard |
| `test_llm_classifier.py` | JSON extraction, category/confidence validation, retries, error results |
| `test_health.py` | Liveness, readiness states, key non-exposure |
| `test_classify_route.py` | Request validation, status-code mapping, error shapes |
| `test_evaluation_metrics.py` | Metric calculations, failure exclusion, echo refusal, error analysis |
| `test_integration_routes.py` | Evaluation endpoint, frontend serving, traversal, secret hygiene |
| `test_cli_entrypoints.py` | Every documented command runs or fails cleanly, in a real subprocess |

Metric correctness is verified against **hand-computed values**, not just
self-consistency — e.g. a known confusion pattern of 7 correct out of 9 must
yield exactly `0.7778` accuracy.

---

## 22. Limitations

**Scope**

- **It does not diagnose disease.** It identifies what kind of information is
  being requested. It never names a likely disease or suggests what is wrong
  with a crop.
- **It does not answer the question.** It routes; it does not advise.
- **It uses no images.** Input is text only. There is no image upload, no vision
  model, and no photo-based identification.
- **It requires no hardware.** No sensors, IoT devices, drones, or field
  equipment are involved.
- **There is no adaptive clarification.** The system does not ask follow-up
  questions or run a multi-turn dialogue. One question in, one classification
  out.

**Technical**

- 300 questions is modest; results may not generalise to other regions,
  languages, or crops.
- Six crops are covered.
- English only.
- Category boundaries are genuinely ambiguous in places — a question can
  reasonably span Prevention and Management. The prompt resolves this by primary
  intent, but some disagreement is irreducible.
- `confidence` is self-reported and uncalibrated.
- Results depend on the chosen model; a different model or version will shift
  the numbers.
- Latency and cost scale with LLM API calls; there is no caching of
  classifications.
- The result page uses `localStorage`, so results do not persist across browsers
  or devices, and the page degrades to an empty state if storage is blocked.
- There is no authentication on the evaluation dashboard.

---

## 23. Future Enhancements

1. **Expand the dataset** — more questions, more crops, more regional phrasing.
2. **Multilingual support** — Tamil, Hindi, and other languages farmers actually
   use, with native-language prompts rather than translation.
3. **Calibrated confidence** — replace the self-reported score with a measured,
   calibrated probability validated against held-out data.
4. **Answer routing** — connect each category to a curated resource library so
   classification leads somewhere useful.
5. **Caching** — deduplicate near-identical questions to cut latency and cost.
6. **Human-in-the-loop correction** — let extension officers flag
   misclassifications and feed them back into the development split.
7. **Dashboard authentication** — protect the evaluation view before any
   public deployment.
8. **Statistical significance testing** — bootstrap confidence intervals on the
   zero-shot/few-shot comparison, so a small delta is not over-read.
9. **Prompt versioning** — record which prompt version produced which results.
10. **Rate limiting** — protect `/api/classify` from abuse in a public
    deployment.

---

## Responsible Use

This tool is a **signpost, not a diagnosis**. It tells a farmer what kind of
information they are asking for. It is not a substitute for a local agronomist,
extension officer, or plant pathologist, and it should never be the basis for a
treatment decision.
