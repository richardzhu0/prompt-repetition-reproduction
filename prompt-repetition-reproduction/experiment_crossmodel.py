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
from google import genai
from dotenv import load_dotenv
from experiment import (
    load_nameindex, summarize, run_mcnemar, run_experiment
)

load_dotenv()

# Model version
MODEL = "gemini-3.8-flash"
# Available: gemini-3.1-flash-lite, gemini-3.5-flash-lite, gemini-3.5-flash, gemini-3.7-flash, gemini-3.8-flash

import experiment
experiment.MODEL = MODEL
experiment.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

if __name__ == "__main__":
    print(f"=== 4.3: CROSS-MODEL COMPARISONS ===")
    print(f"Model: {MODEL}\n")

    print("Loading NameIndex (75 questions)...")
    samples = load_nameindex(n_samples=75)

    print(f"Running NameIndex on {MODEL} (150 calls)...")
    results = run_experiment(samples, is_nameindex=True)
    summarize(results, f"NameIndex — {MODEL}")
    print(f"  McNemar: {run_mcnemar(results)}")

    filename = f"ni_crossmodel_{MODEL.replace('-', '_')}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {filename}")

    print("\n=== ALL DONE ===")