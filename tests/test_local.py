import json
import os
import sys

# Ensure src/ is on Python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.score import init, run

def main():
    # Initialize scoring environment (loads model)
    print("Initializing model...")
    init()

    # Load sample input JSON
    sample_path = os.path.join(os.path.dirname(__file__), "sample_record.json")
    print(f"Loading sample input from {sample_path}...")

    with open(sample_path, "r") as f:
        raw_json = f.read()

    print("Sample input JSON:")
    print(raw_json)

    print("\nRunning prediction...")
    result = run(raw_json)

    print("\n=== Prediction Result ===")
    print(result)


if __name__ == "__main__":
    main()
