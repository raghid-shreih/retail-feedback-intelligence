# Experiment results and interpretation

## Historical notebook observations

The original capstone loaded 23,486 rows and dropped 845 rows with missing review text, leaving 22,641. It sampled 50 reviews with pandas `random_state=42`. These are **recorded notebook outputs**, not results from a fresh run of this package.

| Analysis prompt | Notebook's average LLM-judge score |
| --- | ---: |
| Zero-shot v1 | 0.910 |
| Zero-shot v2 | 0.925 |
| Few-shot v1 | 0.927 |
| Few-shot v2 | 0.877 |
| Stepwise v1 | 0.938 |
| Stepwise v2 | 0.907 |

The notebook also recorded a separate comparative LLM judge favoring zero-shot v2 on 36 of 50 reviews, stepwise v2 on 10, and few-shot v2 on 4. Those comparisons used different scoring and should not be conflated with the averages above. LLM judgments are subjective and not independent human labels.

For recommendation prediction, the notebook reported **46/50 = 92%** on that sample. Its recorded confusion matrix was:

| Actual / predicted | 0 | 1 |
| --- | ---: | ---: |
| 0 | 12 | 0 |
| 1 | 4 | 34 |

The original parser mapped unparseable responses to zero, so this reported accuracy cannot establish how many valid responses were produced. The new evaluator records invalid responses as abstentions and counts them as failures in accuracy across all attempts. Compare new and old accuracy only after checking that all responses were valid and the same data, model, prompts, and parameters were used.

## How to reproduce a new measurement

1. Obtain the source CSV from [Kaggle](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews); record its version and checksum locally.
2. Install the package, set `OPENAI_API_KEY`, and run `retail-feedback recommend` with an explicit model, `--limit 50`, and `--seed 42`.
3. Save the JSON report under ignored `results/`. Record model ID, date, package version, seed, dataset version, valid count, both accuracy denominators, and confusion matrix.
4. For a broader estimate, evaluate an independently selected holdout set, check class balance, and assess errors by subgroup. Prompt comparison reports schema validity only; use independent labels or human reviewers to assess generated content quality.

API behavior and dataset revisions can change. No confidence interval, business uplift, or generalization claim follows from the 50-review notebook sample.
