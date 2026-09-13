import os
import warnings
import logging

warnings.filterwarnings("ignore")
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "3"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
logging.disable(logging.CRITICAL)

import json
import time
from google.genai import types
from dotenv import load_dotenv
from experiment import (
    load_nameindex, extract_nameindex_answer, check_answer, client, MODEL
)

load_dotenv()

def build_repeated_prompt(query: str, n: int) -> str:
    # n value = number of times the query is repeated using the paper's verbose template
    if n == 1:
        return query
    elif n == 2:
        return f"{query}\nLet me repeat that: {query}"
    elif n == 3:
        return f"{query}\nLet me repeat that: {query}\nLet me repeat that one more time: {query}"
    elif n == 4:
        return f"{query}\nLet me repeat that: {query}\nLet me repeat that one more time: {query}\nLet me repeat that for one final time: {query}"
    else:
        raise ValueError(f"Unsupported repetition count: {n}")

def call_model_rep(prompt: str) -> tuple:
    system = "Output only the name. No explanation, no punctuation, no list number, no position, just the name."
    for attempt in range(3):
        try:
            start = time.time()
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=200,
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

def run_repcount_experiment(samples: list, n: int) -> list:
    results = []
    for i, sample in enumerate(samples):
        print(f"  Question {i+1}/{len(samples)}...", end=" ", flush=True)
        query = sample["question"]
        prompt = build_repeated_prompt(query, n)
        text, latency = call_model_rep(prompt)
        answer = extract_nameindex_answer(text)
        correct = check_answer(answer, sample["answer"], is_nameindex=True)
        print(f"rep{n}x='{answer}'({'✓' if correct else '✗'}) correct='{sample['answer']}'")
        results.append({
            "question_id": i,
            "correct_answer": sample["answer"],
            "predicted": answer,
            "correct": correct,
            "latency": latency,
        })
    return results

def summarize_repcount(results: list, label: str):
    n = len(results)
    acc = sum(r["correct"] for r in results) / n
    avg_latency = sum(r["latency"] for r in results) / n
    print(f"\n── {label} ──")
    print(f"  Accuracy: {acc:.1%} ({sum(r['correct'] for r in results)}/{n})")
    print(f"  Avg latency: {avg_latency:.2f}s")
    return acc

if __name__ == "__main__":
    print("=== 4.2: REPETITION COUNT SCALING ===\n")

    print("Loading NameIndex (75 questions)...")
    samples = load_nameindex(n_samples=75)

    REP_COUNT = 4 # what you want the n-value to be (line 23)

    print(f"Running NameIndex with {REP_COUNT}x repetition ({REP_COUNT * 75} calls)...")
    results = run_repcount_experiment(samples, n=REP_COUNT)
    summarize_repcount(results, f"NameIndex {REP_COUNT}x repetition")

    with open(f"ni_rep{REP_COUNT}x_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved ni_rep{REP_COUNT}x_results.json")

    print("\n=== ALL DONE ===")