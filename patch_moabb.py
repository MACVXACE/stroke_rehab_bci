import pathlib, re
p = pathlib.Path(".venv/lib/python3.9/site-packages/moabb/datasets/liu2024.py")
code = p.read_text()
if "on_missing" not in code:
    p.write_text(re.sub(r"(raw\.set_montage\([^\)]+)\)", r"\1, on_missing='ignore')", code))
