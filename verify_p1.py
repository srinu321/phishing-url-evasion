import numpy as np, pandas as pd, json
src=open('p1_phishing.py').read(); exec(src[:src.index('rng = np.random.RandomState(SEED)')])
rng=np.random.RandomState(SEED); pt=Xte[yte==1]
sub=pt.iloc[rng.choice(len(pt),size=800,replace=False)].reset_index(drop=True)
R=json.load(open(D+'results/p1_results.json'))
print("rows raw", len(pd.read_csv(D+'data/phishing_full.csv')), "dedup", len(df), "train/test", len(Xtr), len(Xte), "phish share", round(y.mean(),3))
print("lexical", len(lex), "nonlex", len(nonlex), "overlap", set(lex)&set(nonlex))
print("test phishing n", len(pt), "train/test overlap rows:", len(pd.merge(Xtr.drop_duplicates(), Xte.drop_duplicates())))
m=models()["LogReg"].fit(Xtr,ytr)
p=m.predict_proba(Xte)[:,1]; yh=(p>=.5).astype(int)
print("LogReg F1 recomputed", round(f1_score(yte,yh),4), "stored", round(R['baseline']['LogReg']['clean']['f1'],4))
print("FNR on sub", round((m.predict_proba(sub)[:,1]<.5).mean(),5), "stored k0", R['baseline']['LogReg']['evasion']['0'])
for k in (1,5):
    adv,_=attack(m,sub,k)
    chg=(adv.values!=sub.values)
    print(f"k={k}: max changed/row", chg.sum(1).max(), "nonlex changed", chg[:,[sub.columns.get_loc(c) for c in nonlex]].sum(),
          "evasion recomputed", round((m.predict_proba(adv)[:,1]<.5).mean(),5), "stored", R['baseline']['LogReg']['evasion'][str(k)])
    li=[sub.columns.get_loc(c) for c in lex]
    ok=True
    for j,c in zip(li,lex):
        ch=chg[:,j]
        if ch.any():
            v=adv.values[ch,j]; ok&=bool(((v>=lo[lex.index(c)]-1e-9)&(v<=hi[lex.index(c)]+1e-9)).all())
    print("  all edits within benign p05-p95:", ok)
