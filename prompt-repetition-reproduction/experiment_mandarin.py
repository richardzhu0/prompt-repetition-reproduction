from experiment import (
    run_experiment, summarize, run_mcnemar
)
import random
import json

def load_nameindex_mandarin(n_samples=75):
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
        question = f"以下是一个名字列表：\n{names_str}\n第25个名字是什么？"
        samples.append({
            "question": question,
            "options": None,
            "answer": target
        })
    return samples

if __name__ == "__main__":
    print("=== 4.1: MANDARIN INSTRUCTIONING ===\n")

    print("Loading NameIndex Mandarin (75 questions)...")
    ni_mandarin_samples = load_nameindex_mandarin(n_samples=75)
    print("Running NameIndex Mandarin (150 calls)...")
    ni_mandarin_results = run_experiment(ni_mandarin_samples, is_nameindex=True)
    summarize(ni_mandarin_results, "NameIndex Mandarin")
    print(f"  McNemar: {run_mcnemar(ni_mandarin_results)}")
    with open("ni_mandarin_results.json", "w") as f:
        json.dump(ni_mandarin_results, f, indent=2)
    print("Saved ni_mandarin_results.json")

    print("\n=== ALL DONE ===")