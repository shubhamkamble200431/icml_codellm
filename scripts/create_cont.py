from pathlib import Path
import shutil

SRC_ROOT = Path(
    "/media/kpdubey/8.0 TB Volume/Shubham/github/ICML/icml_codellm/results/"
    "qwen-2.5-coder-7b-instruct/full_dataset_runs"
)

DST_ROOT = Path(
    "/media/kpdubey/8.0 TB Volume/Shubham/github/ICML/icml_codellm/results/"
    "qwen-2.5-coder-7b-instruct/contrastive_runs"
)

DST_ROOT.mkdir(parents=True, exist_ok=True)

copied = 0

for problem_dir in SRC_ROOT.iterdir():

    if not problem_dir.is_dir():
        continue

    has_right = (problem_dir / "right").exists()
    has_wrong = (problem_dir / "wrong").exists()

    if not (has_right and has_wrong):
        continue

    dst_problem = DST_ROOT / problem_dir.name

    if dst_problem.exists():
        shutil.rmtree(dst_problem)

    shutil.copytree(problem_dir, dst_problem)

    copied += 1
    print(f"Copied: {problem_dir.name}")

print(f"\nTotal contrastive problems copied: {copied}")