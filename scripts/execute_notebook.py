"""Execute notebook cells and embed outputs into the .ipynb file."""

import contextlib
import io
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NB_PATH = ROOT / "notebooks" / "severity_experiments.ipynb"


def main():
    with open(NB_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Global execution namespace
    ns = {"__name__": "__main__"}

    execution_count = 1
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue

        code = "".join(cell["source"])
        stdout_buf = io.StringIO()
        cell["outputs"] = []

        try:
            with contextlib.redirect_stdout(stdout_buf):
                # Handle display()
                def custom_display(obj):
                    if hasattr(obj, "to_string"):
                        print(obj.to_string())
                    else:
                        print(str(obj))

                ns["display"] = custom_display

                # Execute
                exec(code, ns)

            text_output = stdout_buf.getvalue()
            if text_output.strip():
                lines = [l + "\n" for l in text_output.splitlines()]
                cell["outputs"].append(
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": lines,
                    }
                )
            cell["execution_count"] = execution_count
            print(f"Executed cell {execution_count} successfully.")
        except Exception as e:
            print(f"Error in cell {execution_count}: {e}")
            cell["outputs"].append(
                {
                    "output_type": "error",
                    "ename": type(e).__name__,
                    "evalue": str(e),
                    "traceback": [str(e)],
                }
            )
        execution_count += 1

    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"Notebook successfully executed and saved to {NB_PATH}")


if __name__ == "__main__":
    main()
