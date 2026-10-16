"""Paper 1: evasion robustness of ML phishing-URL detectors (Vrbancic et al. dataset).
Attacker may only alter URL-lexical features (url/domain/directory/file/params quantities and lengths).
Registration/network features (WHOIS age, ASN, TLS, SPF, DNS, redirects, Google index) are treated as not attacker-controllable.
Greedy black-box coordinate attack, budget k feature edits per sample, edits clipped to the benign 5-95 percentile range.
"""
import re, json, sys, time
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

import sys
SEED = int(sys.argv[1])
MONO = False
import os
D = os.environ.get("P1_ROOT", "./")
df = pd.read_csv(D + "data/phishing_full.csv").drop_duplicates().reset_index(drop=True)
y = df.pop("phishing").values
X = df.replace(-1, 0)  # -1 encodes "not applicable" in this dataset; treat as 0 quantity
cols = list(X.columns)
lex = [c for c in cols if re.search(r"_(url|domain|directory|file|params)$", c) or c in
       ("domain_length", "directory_length", "file_length", "params_length")]
lex = [c for c in lex if c not in ("email_in_url",)]
nonlex = [c for c in cols if c not in lex]
print("features", len(cols), "lexical(controllable)", len(lex), "non-lexical", len(nonlex))

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
Xtr = Xtr.reset_index(drop=True); Xte = Xte.reset_index(drop=True)
ben = Xtr[ytr == 0]
lo, hi = ben[lex].quantile(0.05).values, ben[lex].quantile(0.95).values
med = ben[lex].median().values  # candidate edit target: benign median
cand_vals = {"median": med, "p05": lo, "p95": hi}


def models():
    return {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED),
        "HistGB": HistGradientBoostingClassifier(max_iter=200, random_state=SEED),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=SEED)),
    }


def clean_metrics(m, Xe, ye):
    p = m.predict_proba(Xe)[:, 1]; yh = (p >= 0.5).astype(int)
    return dict(f1=f1_score(ye, yh), prec=precision_score(ye, yh), rec=recall_score(ye, yh), auc=roc_auc_score(ye, p))


def attack(m, Xp, k, rng_idx=None):
    """Greedy coordinate search. Returns adversarial copy of Xp (phishing rows), k edits per row max."""
    A = Xp.copy().reset_index(drop=True)
    L = len(lex); n = len(A)
    used = np.zeros((n, L), dtype=bool)
    lex_idx = [A.columns.get_loc(c) for c in lex]
    base = A.values.astype(float)
    cur = m.predict_proba(A)[:, 1]
    traj = {0: float((cur < 0.5).mean())}
    for step in range(k):
        best_gain = np.zeros(n); best_j = -np.ones(n, dtype=int); best_v = np.zeros(n)
        # evaluate every (feature, candidate value) edit
        for name, vals in cand_vals.items():
            for j in range(L):
                trial = base.copy()
                trial[:, lex_idx[j]] = vals[j]
                pr = m.predict_proba(pd.DataFrame(trial, columns=A.columns))[:, 1]
                gain = cur - pr
                if MONO: gain[~(vals[j] > base[:, lex_idx[j]])] = -1
                gain[used[:, j]] = -1
                upd = gain > best_gain
                best_gain = np.where(upd, gain, best_gain)
                best_j = np.where(upd, j, best_j)
                best_v = np.where(upd, vals[j], best_v)
        ok = best_j >= 0
        for i in np.where(ok)[0]:
            base[i, lex_idx[best_j[i]]] = best_v[i]; used[i, best_j[i]] = True
        cur = m.predict_proba(pd.DataFrame(base, columns=A.columns))[:, 1]
        traj[step + 1] = float((cur < 0.5).mean())
    return pd.DataFrame(base, columns=A.columns), traj


def evasion_curve(m, Xp, kmax=10):
    A, traj = attack(m, Xp, kmax)
    return {k: traj[k] for k in KS}


rng = np.random.RandomState(SEED)
phish_te = Xte[yte == 1]
sub = phish_te.iloc[rng.choice(len(phish_te), size=800, replace=False)].reset_index(drop=True)
KS = [0, 1, 3, 5, 10]
out = {"seed": SEED, "unconstrained": {}, "addonly": {}}
import time; t0 = time.time()
fn = D + f"results/p1d_seed{SEED}.json"
for name, m in models().items():
    m.fit(Xtr, ytr)
    cm = clean_metrics(m, Xte, yte)
    MONO = False; e1 = evasion_curve(m, sub)
    out["unconstrained"][name] = {"clean": cm, "evasion": e1}
    MONO = True; e2 = evasion_curve(m, sub)
    out["addonly"][name] = {"evasion": e2}
    print(SEED, name, round(cm["f1"], 4), e1, e2, f"{time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(fn, "w"), indent=1)
print("done")
