"""Write a fixed random BFCL subset (seed 0, N per category) to $BFCL_PROJECT_ROOT/test_case_ids_to_generate.json."""
import json, os, random, sys
from pathlib import Path
import bfcl_eval

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
CATS = ["simple_python", "multiple", "parallel_multiple", "irrelevance", "live_simple", "live_multiple", "multi_turn_base"]
data = Path(bfcl_eval.__file__).parent / "data"
ids = {}
for c in CATS:
    rows = [json.loads(l)["id"] for l in (data / f"BFCL_v4_{c}.json").open()]
    ids[c] = sorted(random.Random(0).sample(rows, min(N, len(rows))))
out = Path(os.environ["BFCL_PROJECT_ROOT"]) / "test_case_ids_to_generate.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(ids, indent=1))
print({c: len(v) for c, v in ids.items()})
