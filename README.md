# Phishing URL detector evasion study: code and results

Code and result files for the manuscript *How Robust Are Machine-Learning Phishing URL Detectors to Cheap Feature Edits? An Empirical Evasion Study with Controllable-Feature Threat Models and Defenses*.

**Author:** Srinivas Goud Thadakapally (ORCID 0009-0006-2505-0807)

## Data
Vrbančič et al., *Phishing Websites Dataset* (Mendeley Data, doi:10.17632/72ptz43s9v.1). Download the full dataset (88,647 rows) and save it as `data/phishing_full.csv` (label column `phishing`). The data are not redistributed here.

## Requirements
Python 3.11, numpy, pandas, scikit-learn (`pip install -r requirements.txt`).

## Running
Run from the repository folder with the environment variable `P1_ROOT=./` (trailing slash). The scripts read `data/` and write `results/`.

1. `python p1_phishing.py`: baseline, non-lexical-only defense and adversarial training; writes `results/p1_results.json`.
2. `python p1b_monotone.py`: add-only attack for logistic regression and random forest (`python p1b_one.py HistGB` and `python p1b_one.py MLP` for the other two models).
3. `python p1c_adv.py RandomForest` and `python p1c_adv.py MLP`: adversarial training for those two models.
4. `python p1d_seeds.py SEED` (SEED = 1 to 4): repeats the baseline and add-only attacks on a new split and sample.
5. `python verify_p1.py`: independent re-check of the attack constraints and one model's numbers.

The main run uses seed 7. The attack is slow for random forest, gradient boosting and the MLP (minutes to tens of minutes on 2 CPUs).

`p1_phishing.py` was run to completion for the baselines and the non-lexical defense, and for adversarial training of logistic regression and gradient boosting; the other two adversarial-training runs were done with `p1c_adv.py`. The result files in `results/` are the outputs reported in the manuscript.

## License
MIT (see `LICENSE`).
