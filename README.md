# prompt-repetition-reproduction

A reproduction of Leviathan, Kalman, and Matias **"Prompt Repetition Improves Non-Reasoning LLMs"** ([arXiv:2512.14982] (https://arxiv.org/abs/2512.14982)) on six current Gemini models. 

**The paper's claim still holds true.** Repeating the prompt verbatim produced large, statistically significant accuracy gains on the NameIndex positional-retrieval task, with no latency penalty. ARC-Challenge results were directionally consistent but did not reach significance at n=100.

A failure taxonomy is included to classify wrong answers into various categories, which shows the effect arrives by different routes on different model families. On lite models, repetition removes ordinary wrong answers and nothing else. On flash models, most baseline failures come with malformed output that runs past the answer into the next list item, and repetition suppresses malformed output and the answers together. 

The failure taxonomy, as well as the per-question results for every run, and the Python code can be found in the attached folder (prompt-repetition-reproduction).

**[Full report](report.pdf)**
