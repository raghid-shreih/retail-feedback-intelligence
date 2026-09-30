# Retail Feedback Intelligence

A reproducible portfolio project for analyzing apparel reviews and testing whether a language model can predict the reviewer's product recommendation. It turns the original exploratory notebook into a small Python package with bounded command line experiments, schema validation, offline tests, and a clean walkthrough notebook.

## What it does

- Loads the [Women's E-Commerce Clothing Reviews dataset](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews), including comma-separated source files and semicolon-separated notebook exports.
- Compares six original analysis prompt variants (zero-shot, few-shot, and stepwise prompts), recording valid structured responses. Schema validity alone does **not** establish answer quality.
- Evaluates a binary recommendation prompt against the dataset's `Recommended IND` label. Invalid responses abstain and count as failed attempts in `accuracy_all_attempts`; the confusion matrix includes valid predictions only.

## Setup

Python 3.10+ is required. Download the CSV from the Kaggle page above and place it under `data/` (which is ignored by Git). The dataset is **not distributed here**; check its source terms before using or redistributing it. No API key is needed for the offline inspection or tests.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
retail-feedback summary data/Womens\ Clothing\ E-Commerce\ Reviews.csv
python -m unittest discover -s tests -v
```

For a small, explicit live experiment, set `OPENAI_API_KEY` in your shell and choose an available chat model that supports JSON response formatting. The default sample is five reviews; each analysis variant makes one request per review. API calls may incur charges. Keep keys out of source files and outputs.

```bash
export OPENAI_API_KEY="your-key"
retail-feedback recommend data/Womens\ Clothing\ E-Commerce\ Reviews.csv --model YOUR_MODEL_ID --limit 5 --output results/recommendations.json
retail-feedback compare data/Womens\ Clothing\ E-Commerce\ Reviews.csv --model YOUR_MODEL_ID --limit 5 --variants zero_v1 zero_v2 few_v1 few_v2 cot_v1 cot_v2 --output results/comparison.json
```

The source file may have a different downloaded name; pass its actual path. Sampling uses pandas with seed 42 by default (`--seed` changes it). `data/`, `results/`, `.env`, and notebook outputs are excluded from version control.

## Results and limits

The [results note](docs/results.md) separates historical notebook observations from a reproducible rerun. The original notebook's 92% recommendation accuracy was measured on 50 sampled reviews, using an LLM to classify them. It is an exploratory result, not an out-of-sample validation or a claim about production accuracy. No paid API evaluation was run to prepare this repository.

The six prompts are adapted from the original notebook. They contain the fictional retail brand names and wording used in that experiment; minor inconsistencies in the original prompts are preserved so the variants remain recognizable. The new code uses the official OpenAI SDK endpoint by default and does not depend on a course-specific proxy or Google Drive mount.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/retail_feedback/` | Data loading, original prompts, validation, evaluation, CLI |
| `notebooks/walkthrough.ipynb` | Clean offline walkthrough with a guarded optional API example |
| `tests/` | Synthetic fixture and offline workflow tests |
| `docs/results.md` | Historical findings, limitations, and reproducibility notes |

## Responsible use

Customer replies and retail insights are generated drafts; review them before sending or acting. The dataset contains customer-written text, so avoid publishing raw reviews or API responses without checking the dataset's terms and privacy implications. The CSV and generated reports stay local by default.

Code license: [MIT](LICENSE). The dataset and model services have separate terms.
