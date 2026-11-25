# Fine-tuned small model vs. a bigger prompted model: banking complaint routing

For a narrow, high-volume banking task, can a small open model fine-tuned with QLoRA match or beat a larger, well-prompted model on accuracy? And what does each one cost per 1,000 complaints?

**Task:** route a customer complaint about a checking or savings account to its *issue* category. The data is real complaint narratives from the public [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/).

## Results

All three systems were scored on the same held-out test set of 400 complaints. The fine-tuned model was trained on 3,502 examples.

| System | Model | Accuracy | Macro F1 (95% CI) | Sec / item | Cost / 1k complaints |
| --- | --- | --- | --- | --- | --- |
| `small_base_fewshot` | Qwen2.5-1.5B-Instruct, few-shot prompt | 0.178 | 0.095 (0.070–0.122) | 0.47 | $0.068 |
| `large_open_fewshot` | Qwen2.5-7B-Instruct, few-shot prompt | 0.348 | 0.232 (0.194–0.271) | 1.55 | $0.226 |
| **`small_finetuned`** | **Qwen2.5-1.5B-Instruct + QLoRA** | **0.670** | **0.418 (0.388–0.445)** | **0.21** | **$0.030** |

- The fine-tuned 1.5B model beats the 7B few-shot baseline by a wide margin, and the confidence intervals don't overlap. It's also about 7× faster per item and about 7× cheaper.
- Training took 24.7 minutes on one T4 (peak GPU memory 11.4 GB). That's about $0.22 at $0.526/hour, so the one-time training cost is paid back after roughly 1,000 complaints.
- No system produced invalid (unparseable) outputs.
- Cost assumes an AWS g4dn.xlarge (1× T4) at $0.526/hour on demand. Small-model speeds are batched throughput.

### Per-class F1

| Issue | 7B few-shot | 1.5B few-shot | 1.5B fine-tuned | Test support |
| --- | --- | --- | --- | --- |
| Managing an account | 0.263 | 0.090 | 0.465 | 78 |
| Closing an account | 0.253 | 0.159 | 0.685 | 78 |
| Problem with a lender or other company charging your account | 0.556 | 0.121 | 0.684 | 78 |
| Opening an account | 0.227 | 0.050 | 0.772 | 78 |
| Problem caused by your funds being low | 0.416 | 0.288 | 0.742 | 78 |
| Incorrect information on your report | 0.143 | 0.056 | 0.000 | 4 |
| Problem with a company's investigation into an existing problem | 0.000 | 0.000 | 0.000 | 3 |
| Improper use of your report | 0.000 | 0.000 | 0.000 | 3 |

The last three issues are rare in this product: they have only 3–4 test examples each. They pull macro F1 down for every system, which is why macro F1 sits well below accuracy. Dropping them, or lowering `TOP_K_ISSUES`, would give a cleaner comparison.

Confusion matrices for the fine-tuned model and the baseline are in [results/confusion_matrices.png](results/confusion_matrices.png).

### Caveats

- **Noisy labels:** consumers pick the issue category themselves when they file a complaint, which limits how accurate any system can be.
- One product, one random split, US data only.
- The baseline in this run is a 7B open model running locally, not a frontier API model. The notebook can also run Gemini or Claude as the baseline (see [Configuration](#configuration)).

## How it works

[finetune-vs-frontier.ipynb](finetune-vs-frontier.ipynb) runs the whole experiment from top to bottom:

1. **Data:** downloads two CFPB FOIA archive ZIPs (Nov 2022 – Mar 2024). It keeps checking/savings complaints from 2023 onward that have a narrative of at least 50 characters, removes duplicates, keeps the 8 most common issues (at most 800 complaints each), and makes a stratified split: train, 200 for validation, 400 for test.
2. **Prompts:** a system prompt plus the list of labels. The few-shot prompts add one example per label, taken from the training set. A parser maps the model's output back to a label; anything it can't map counts as invalid.
3. **Baseline A:** the bigger model with a few-shot prompt. This is a local 7B model, Gemini, or Claude, depending on the `BASELINE` setting.
4. **Baseline B:** the small model with the same few-shot prompt and no training.
5. **QLoRA fine-tune:** the small model loaded in 4-bit NF4 with LoRA adapters (r=16, α=32, dropout 0.05) on every attention and MLP projection. Training uses lr 2e-4 for 1 epoch with an effective batch size of 16. Loss is computed only on the label tokens of the reply.
6. **Evaluation:** accuracy, macro F1 with a bootstrap 95% CI, per-class F1, confusion matrices, and error analysis (cases where only one system was right, or both were wrong).

## Running it

You need a CUDA GPU. A Colab or Kaggle T4 is enough. Run the notebook from top to bottom. It installs its own dependencies:

```
transformers peft accelerate bitsandbytes datasets scikit-learn anthropic
```

Add `google-genai` if you use the Gemini baseline.

Expect about 10 minutes for the data download and filtering, about 25 minutes for fine-tuning, and a few minutes for each evaluation. Predictions are cached in `results/`, so a disconnect doesn't lose finished steps. The filtered data is cached as `data/archive_checking_or_savings_account.parquet`.

### Configuration

Settings are in section 1 of the notebook:

| Setting | Default | Notes |
| --- | --- | --- |
| `SMALL_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Try `Qwen/Qwen2.5-3B-Instruct` |
| `BASELINE` | `local_large` | `local_large`, `gemini`, or `anthropic` |
| `LARGE_LOCAL_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Used when `BASELINE="local_large"` |
| `TOP_K_ISSUES`, `MAX_PER_CLASS` | 8, 800 | Label set and class balance |
| `GPU_PRICE_PER_HOUR`, `FRONTIER_PRICE_*` | 0.526, 1.50 / 7.50 | Used for the cost columns |

API keys (`GEMINI_API_KEY` or `ANTHROPIC_API_KEY`) are read from Colab secrets, then from environment variables. If neither has them, the notebook asks for them.

## Repository layout

```
finetune-vs-frontier.ipynb   # the full experiment
results/                     # committed: metrics, per-class F1, predictions, confusion matrices, run config
data/                        # git-ignored: CFPB ZIPs (~228 MB) + filtered parquet cache
outputs/lora_adapter/        # git-ignored: trained LoRA adapter (~71 MB)
```

The notebook recreates `data/` and `outputs/`.

## Next experiments

- A 3B small model, more training data, two epochs
- Distillation: train on frontier-model labels for unlabelled complaints
- 8-bit vs. 4-bit inference
- A frontier API baseline (Gemini or Claude) instead of the local 7B model
