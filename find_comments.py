import os
import glob

files = []
files.extend(glob.glob("*.py"))
files.extend(glob.glob("*.sh"))
files.extend(glob.glob("scripts/*.py"))
files.extend(glob.glob("models/*.py"))
files.extend(glob.glob("data/*.py"))

ignore = ["patch_", "test_", "debug_", "fix_", "envcheck", "run_make"]
for f in files:
    if any(ign in f for ign in ignore):
        continue
    with open(f, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if '#' in line:
                print(f"{f}:{i+1}:{line.strip()}")
EOF
python3 find_comments.py

