import os
import re
import time
import json
import warnings
import logging

# Warning suppressions
warnings.filterwarnings("ignore")
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "3"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
logging.disable(logging.CRITICAL)

import random
from datasets import load_dataset
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-3.6-flash" # Primary experiment model; gemini-3.6-flash

# §1 Prompt building
def build_query(question: str, options: dict, ordering: str) -> str:
    options_str = "\n".join(f"{k}. {v}" for k, v in options.items())
    format_instruction = "Reply with one letter ('A', 'B', 'C', 'D') in the format:\nThe answer is <ANSWER>."
    if ordering == "question-first":
        return f"{question}\n{options_str}\n{format_instruction}"
    else:
        return f"{options_str}\n{question}\n{format_instruction}"

def build_baseline_prompt(query: str) -> str:
    return query

def build_repetition_prompt(query: str) -> str:
    return f"{query}\n{query}"

# §2 Loading datasets (ARC + NameIndex)

def load_arc(n_samples=100):
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    dataset = dataset.shuffle(seed=42).select(range(n_samples))
    samples = []
    for item in dataset:
        options = {}
        for label, text in zip(item["choices"]["label"], item["choices"]["text"]):
            options[label] = text
        samples.append({
            "question": item["question"],
            "options": options,
            "answer": item["answerKey"]
        })
    return samples

def load_nameindex(n_samples=50):
    first_names = [
        "James", "John", "Robert", "Michael", "William", "David", "Richard",
        "Joseph", "Thomas", "Charles", "Daniel", "Matthew", "Anthony", "Mark",
        "Donald", "Steven", "Paul", "Andrew", "Kenneth", "Joshua", "Mary",
        "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth", "Susan",
        "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty", "Margaret",
        "Sandra", "Ashley", "Dorothy", "Kimberly", "Emily", "Donna"
    ]
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
        "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
        "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
        "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
        "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
        "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores"
    ]
    samples = []
    random.seed(42)
    for _ in range(n_samples):
        names = [f"{random.choice(first_names)} {random.choice(last_names)}"
                 for _ in range(50)]
        target = names[24]
        names_str = ", ".join(names)
        question = f"Here's a list of names:\n{names_str}\nWhat's the 25th name?"
        samples.append({
            "question": question,
            "options": None,
            "answer": target
        })
    return samples

# §3 Running

def extract_answer(response_text: str) -> str:
    if not response_text or response_text in ("BLOCKED", "ERROR"):
        return "NONE"
    # Pre-stage
    cleaned_text = re.sub(r'[*_`"]', '', response_text)
    # Stage 1
    stage1_pattern = r'(?:the\s+answer\s+is|answer|option|choice)\s*:?\s*([A-D])\b'
    matches = re.findall(stage1_pattern, cleaned_text, re.IGNORECASE)
    if matches:
        return matches[-1].upper() # Most LLMs provide the final answer at the end.
    # Stage 2
    stage2_match = re.search(r'^\s*([A-D])\s*[.)]?\s*$', cleaned_text, re.MULTILINE)
    if stage2_match:
        return stage2_match.group(1).upper()
    # Stage 3
    letters = re.findall(r'\b([A-D])\b', cleaned_text)
    if letters:
        # Filter out lowercase 'a' or accidental matches if original was lowercase
        return letters[-1].upper()
    return "NONE"

def extract_nameindex_answer(response_text: str) -> str:
    if not response_text or response_text in ("BLOCKED", "ERROR"):
        return "NONE"
    first_line = response_text.split('\n')[0].strip()
    first_line = re.sub(r'^[\d.)\s,:;]+', '', first_line)
    return first_line.strip()

def call_model(prompt: str, is_nameindex: bool = False) -> tuple:
    system = "Output only the name. No explanation, no punctuation, no list number, no position, just the name." if is_nameindex else "Do not explain your reasoning. Answer directly and concisely."
    max_tokens = 200 if is_nameindex else 150
    for attempt in range(3):
        try:
            start = time.time()
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=max_tokens,
                    system_instruction=system,
                )
            )
            latency = time.time() - start
            try:
                text = response.text.strip()
                if not text:
                    text = "BLOCKED"
            except Exception:
                text = "BLOCKED"
            return text, latency
        except Exception as e:
            time.sleep(5)
    return "ERROR", 0.0

def check_answer(predicted: str, correct: str, is_nameindex: bool = False) -> bool:
    if is_nameindex:
        return predicted.strip().lower() == correct.strip().lower()
    return predicted.upper() == correct.upper()

def run_experiment(samples: list, ordering: str = "question-first",
                   is_nameindex: bool = False, n_samples: int | None = None) -> list:
    if n_samples:
        samples = samples[:n_samples]
    results = []
    for i, sample in enumerate(samples):
        print(f"  Question {i+1}/{len(samples)}...", end=" ", flush=True)

        if is_nameindex:
            query = sample["question"]
        else:
            query = build_query(sample["question"], sample["options"], ordering)

        baseline_text, baseline_latency = call_model(
            build_baseline_prompt(query), is_nameindex)
        rep_text, rep_latency = call_model(
            build_repetition_prompt(query), is_nameindex)

        if is_nameindex:
            baseline_answer = extract_nameindex_answer(baseline_text)
            rep_answer = extract_nameindex_answer(rep_text)
        else:
            baseline_answer = extract_answer(baseline_text)
            rep_answer = extract_answer(rep_text)

        baseline_correct = check_answer(baseline_answer, sample["answer"], is_nameindex)
        rep_correct = check_answer(rep_answer, sample["answer"], is_nameindex)

        print(f"baseline='{baseline_answer}'({'✓' if baseline_correct else '✗'}) "
              f"rep='{rep_answer}'({'✓' if rep_correct else '✗'}) "
              f"correct='{sample['answer']}'")

        results.append({
            "question_id": i,
            "correct_answer": sample["answer"],
            "baseline_raw": baseline_text,
            "baseline_answer": baseline_answer,
            "baseline_correct": baseline_correct,
            "baseline_latency": baseline_latency,
            "repetition_raw": rep_text,
            "repetition_answer": rep_answer,
            "repetition_correct": rep_correct,
            "repetition_latency": rep_latency,
        })
    return results

def summarize(results: list, label: str):
    n = len(results)
    baseline_acc = sum(r["baseline_correct"] for r in results) / n
    rep_acc = sum(r["repetition_correct"] for r in results) / n
    print(f"\n── {label} ──")
    print(f"  Baseline accuracy:    {baseline_acc:.1%} ({sum(r['baseline_correct'] for r in results)}/{n})")
    print(f"  Repetition accuracy:  {rep_acc:.1%} ({sum(r['repetition_correct'] for r in results)}/{n})")
    print(f"  Difference:           {rep_acc - baseline_acc:+.1%}")
    avg_baseline_latency = sum(r["baseline_latency"] for r in results) / n
    avg_rep_latency = sum(r["repetition_latency"] for r in results) / n
    print(f"  Avg baseline latency: {avg_baseline_latency:.2f}s")
    print(f"  Avg repetition latency: {avg_rep_latency:.2f}s")
    print(f"  Latency difference:   {avg_rep_latency - avg_baseline_latency:+.2f}s")

# §4 McNemar test

def run_mcnemar(results: list) -> dict:
    from statsmodels.stats.contingency_tables import mcnemar
    baseline = [r["baseline_correct"] for r in results]
    repetition = [r["repetition_correct"] for r in results]
    b = sum(1 for a, r in zip(baseline, repetition) if a and not r)
    c = sum(1 for a, r in zip(baseline, repetition) if not a and r)
    n_both_correct = sum(1 for a, r in zip(baseline, repetition) if a and r)
    n_both_wrong = sum(1 for a, r in zip(baseline, repetition) if not a and not r)
    table = [[n_both_correct, b], [c, n_both_wrong]]
    result = mcnemar(table, exact=True)
    return {
        "b (baseline_only_correct)": b,
        "c (rep_only_correct)": c,
        "p_value": round(result.pvalue, 4),
        "significant_at_0.1": result.pvalue < 0.1
    }

# §5 full experiment!

if __name__ == "__main__":
    print("=== FULL EXPERIMENT ===\n")

    print("Loading ARC (100 questions)...")
    arc_samples = load_arc(n_samples=100)

    print("Running ARC question-first (200 calls)...")
    arc_qf = run_experiment(arc_samples, ordering="question-first")
    summarize(arc_qf, "ARC question-first")
    print(f"  McNemar: {run_mcnemar(arc_qf)}")
    with open("arc_qf_results.json", "w") as f:
        json.dump(arc_qf, f, indent=2)
    print("Saved arc_qf_results.json")

    print("\nRunning ARC options-first (200 calls)...")
    arc_of = run_experiment(arc_samples, ordering="options-first")
    summarize(arc_of, "ARC options-first")
    print(f"  McNemar: {run_mcnemar(arc_of)}")
    with open("arc_of_results.json", "w") as f:
        json.dump(arc_of, f, indent=2)
    print("Saved arc_of_results.json")

    print("\nLoading NameIndex (75 questions)...")
    ni_samples = load_nameindex(n_samples=75)
    print("Running NameIndex (150 calls)...")
    ni_results = run_experiment(ni_samples, is_nameindex=True)
    summarize(ni_results, "NameIndex")
    print(f"  McNemar: {run_mcnemar(ni_results)}")
    with open("ni_results.json", "w") as f:
        json.dump(ni_results, f, indent=2)
    print("Saved ni_results.json")

    print("\n=== ALL DONE ===")