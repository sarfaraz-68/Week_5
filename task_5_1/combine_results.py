import json
import glob

files = sorted(
    glob.glob("evaluation_result_*.json"),
    key=lambda x: int(
        x.split("_")[-1].split(".")[0]
    )
)

all_results = []

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        result = json.load(f)

    all_results.append(result)

with open(
    "evaluation_results.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        all_results,
        f,
        indent=4,
        ensure_ascii=False
    )

print("Combined evaluation results:")
print(f"Total tests: {len(all_results)}")

passed = sum(
    1 for result in all_results
    if result["pass"]
)

print(f"Passed: {passed}")
print(f"Failed: {len(all_results) - passed}")

print("\nSaved: evaluation_results.json")