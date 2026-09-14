# prompt-repetition-reproduction

A reproduction of Leviathan, Kalman, and Matias **"Prompt Repetition Improves Non-Reasoning LLMs"** ([arXiv:2512.14982](https://arxiv.org/abs/2512.14982)) on six current Gemini models. 

**[Read the full report](report.pdf)**

## What was found

**The paper's claim still holds true.** Repeating the prompt verbatim produced large, statistically significant accuracy gains on the NameIndex positional-retrieval task, with no latency penalty. ARC-Challenge results were directionally consistent but did not reach significance at n=100.

A failure taxonomy is included to classify wrong answers into various categories, which shows the effect arrives by different routes on different model families. On lite models, repetition removes ordinary wrong answers and nothing else. On flash models, most baseline failures come with malformed output that runs past the answer into the next list item, and repetition suppresses malformed output and the answers together. 

**gemini-3.5-flash actually lost accuracy under repetition**, which is a counterexample to the paper's record of 47 wins and zero losses.

---

## Results

### ARC Challenge (n=100)

| Ordering | Baseline | Repetition | Δ | p |
|---|---:|---:|---:|---:|
| Question-first | 87.00% | 93.00% | +6.00 | 0.1094 |
| Options-first | 84.00% | 90.00% | +6.00 | 0.2632 |

### NameIndex (mean of three runs, n=75)

| Model | Baseline | Repetition | Δ |
|---|---:|---:|---:|
| gemini-2.0-flash-lite *(paper)* | 21.33% | 97.33% | +76.00% |
| gemini-3.1-flash-lite | 70.67% | 100.00% | +29.33% |
| gemini-3.5-flash-lite | 45.33% | 95.56% | +50.22% |
| gemini-3.5-flash | 85.78% | 81.78% | −4.00% |
| gemini-3.6-flash | 68.89% | 97.78% | +28.89% |
| gemini-3.7-flash | 80.00% | 94.22% | +14.22% |
| gemini-3.8-flash | 82.22% | 95.56% | +13.33% |

### Repetition count scaling (gemini-3.6-flash)

| Repetitions | Template | Mean accuracy |
|---|---|---:|
| 1× | none | 68.89% |
| 2× | `<Query>\n<Query>` | 97.78% |
| 2× verbose | paper's template | 95.08% |
| 3× verbose | paper's template | 94.22% |
| 4× verbose | custom | 89.72% |

A bare newline outperformed every verbose variant, which does not substantiate the paper's claim that 3× substantially outperforms 2×.


### Mandarin instructions (gemini-3.6-flash)

| Condition | Baseline | Repetition |
|---|---:|---:|
| English | 68.89% | 97.78% |
| Mandarin | 70.67% | 93.78% |

### Latency (424 baseline / 425 repetition calls)

| | Median | Mean |
|---|---:|---:|
| Baseline | 2.60s | 5.99s |
| Repetition | 2.54s | 5.44s |

---

## The failure taxonomy

Every failure is assigned to one of five categories rather than scored as simply incorrect.

| Category | Definition |
|---|---|
| `ERROR` | API call did not complete; excluded from the denominator of 75 |
| `BLOCKED` | Provider returned a block instead of generated text |
| `EMPTY` | Model returned text, extractor found no name |
| `LEAKED` | Model answered, then continued into the next list item |
| `OTHER` | A single well-formed but incorrect name |

Two hypothesis of `LEAKED` were tested and rejected:

- **Does the extractor discard correct answers?** No. Across 312 non-infrastructure failures (failures excluding `ERROR` and `BLOCKED`), the correct answer appeared anywhere in the raw output exactly once.
- **Do index-prefixed responses like `23. Mary Lewis` mean the model answered a different list position?** No. Of 44 such responses, none contained the correct answer for the index named in the prefix, or either adjacent index.

The failure taxonomy, as well as the per-question results for every run, and the Python code can be found in the attached folder (prompt-repetition-reproduction).

**[Read the full report](report.pdf)**
