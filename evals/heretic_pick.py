"""Pick the Heretic trial to export: fewest refusals with KL divergence <= MAX_KL.
Prints the Pareto front to stderr and the chosen index (in Heretic's own sorted-front order) to stdout.
With SAVE_DIR, also rewrites the settings stored in the study so the next `heretic` run with
checkpoint_action = "continue" runs no new trials and exports that trial merged to SAVE_DIR
(on continue, Heretic replaces every setting with the stored ones, so config.toml can't do this).
HERETIC_TRIAL=<n> overrides the rule and picks trial n (it must be on the Pareto front).
usage: python heretic_pick.py checkpoints/<model>.jsonl [SAVE_DIR]
"""
import json
import os
import sys

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

MAX_KL = 0.2  # training afterwards heals some drift, so a bit looser than Heretic's usual ~0.1

study = optuna.load_study(study_name="heretic", storage=JournalStorage(JournalFileBackend(sys.argv[1])))


def score(trial, name):
    return next(s["score"]["value"] for s in trial.user_attrs["scores"] if s["name"] == name)


# Same order as Heretic's trial menu: by each objective, ascending.
front = sorted(study.best_trials, key=lambda t: (score(t, "Refusals"), score(t, "KL divergence")))
for i, t in enumerate(front):
    print(f"front[{i}] trial {t.user_attrs['index']}: refusals {score(t, 'Refusals')}, "
          f"KL {score(t, 'KL divergence'):.4f}", file=sys.stderr)
ok = [i for i, t in enumerate(front) if score(t, "KL divergence") <= MAX_KL]
k = ok[0] if ok else len(front) - 1  # front is refusal-sorted, so the first ok has fewest refusals
if os.environ.get("HERETIC_TRIAL"):
    k = next(i for i, t in enumerate(front) if t.user_attrs["index"] == int(os.environ["HERETIC_TRIAL"]))
print(f"chosen: front[{k}] = trial {front[k].user_attrs['index']}", file=sys.stderr)

if len(sys.argv) > 2:
    s = json.loads(study.user_attrs["settings"])
    s.update(n_trials=len(study.trials), trial_index=k, export_strategy="merge",
             model_action="save", save_directory=sys.argv[2])
    study.set_user_attr("settings", json.dumps(s))
print(k)
