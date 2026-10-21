"""Generate the 8 PCA graphs + results.json.  Run from the project root:  python generate_graphs.py
Data: data/breast_cancer_wisconsin.csv (UCI WDBC, 569 samples; see README for provenance)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

SEED = 42; np.random.seed(SEED)
CSV = Path("data/breast_cancer_wisconsin.csv"); OUT = Path("assets/images"); OUT.mkdir(parents=True, exist_ok=True)
if not CSV.exists(): sys.exit(f"Missing {CSV}. Download WDBC from UCI (https://archive.ics.uci.edu/dataset/17) and convert to CSV, see README.")
df = pd.read_csv(CSV)
if "diagnosis" not in df.columns: sys.exit("Expected a 'diagnosis' column (M/B).")
ID_COLS = [c for c in ["id"] if c in df.columns]                 # identifier (absent in the bundled copy)
FEATURES = [c for c in df.columns if c not in ID_COLS + ["diagnosis"]]   # 30 numeric measurements
if not all(np.issubdtype(df[c].dtype, np.number) for c in FEATURES): sys.exit("Non-numeric feature found.")
print("shape", df.shape, "| features", len(FEATURES), "| missing", int(df[FEATURES].isna().sum().sum()))
assert df[FEATURES].isna().sum().sum() == 0, "Handle missing values (e.g. impute on training data) before PCA."
y = df["diagnosis"].to_numpy(); n, p = len(df), len(FEATURES)
sns.set_theme(style="whitegrid", font_scale=1.0)
COL = {"B": "#1f77b4", "M": "#d62728"}
R = {"n": n, "p": p, "features": FEATURES, "missing": 0, "class_counts": df.diagnosis.value_counts().to_dict(), "seed": SEED}
def save(fig, name): fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---- Preprocessing + PCA on the analysis data (unsupervised; diagnosis NOT used) ----
Xs = StandardScaler().fit_transform(df[FEATURES])
pca = PCA(random_state=SEED).fit(Xs)
Z = pca.transform(Xs); evr = pca.explained_variance_ratio_; cum = np.cumsum(evr); lam = pca.explained_variance_
R.update(evr=evr.tolist(), cum=cum.tolist(), eigenvalues=lam.tolist(),
         k90=int(np.argmax(cum >= .90) + 1), k95=int(np.argmax(cum >= .95) + 1), k99=int(np.argmax(cum >= .99) + 1))

# ---- Graph 1: two original features ----
a, b = "mean radius", "mean texture"
fig, ax = plt.subplots(figsize=(8, 5.5))
for c, lab in [("B", "Benign (B)"), ("M", "Malignant (M)")]:
    m = y == c; ax.scatter(df[a][m], df[b][m], s=26, alpha=.7, color=COL[c], label=lab, edgecolor="white", lw=.4)
ax.set(xlabel="Mean radius (mean distance from centre to nucleus boundary; image units)",
       ylabel="Mean texture (std. dev. of grey-scale values)", title="Two of 30 original features (n = 569)")
ax.legend(title="Diagnosis (colour only; not used by PCA)")
ax.text(.02, .02, "This 2-D view shows 2 of 30 features,\nonly a small slice of the data.", transform=ax.transAxes, fontsize=9,
        bbox=dict(fc="#f1f5f9", ec="#cbd5e1", boxstyle="round"))
save(fig, "pca-original-feature-distribution.png")

# ---- Graph 2: correlation heatmap of the 10 'mean' features ----
mean_cols = [c for c in FEATURES if c.startswith("mean ")]
corr = df[mean_cols].corr()
fig, ax = plt.subplots(figsize=(9.5, 7.5))
sns.heatmap(corr, annot=True, fmt=".2f", annot_kws={"size": 8}, cmap="coolwarm", vmin=-1, vmax=1, square=True,
            cbar_kws={"label": "Pearson correlation"}, ax=ax)
ax.set_title("Correlations among the 10 'mean' features"); plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
save(fig, "pca-feature-correlation-heatmap.png")
tri = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
R["corr"] = {"radius_perimeter": float(corr.loc["mean radius", "mean perimeter"]), "radius_area": float(corr.loc["mean radius", "mean area"]),
             "radius_texture": float(corr.loc["mean radius", "mean texture"]), "min_pair": [list(tri.idxmin()), float(tri.min())],
             "n_abs_gt_09": int((tri.abs() > .9).sum()), "n_pairs": int(len(tri))}

# ---- Graph 3: before/after rotation on 150 standardised points, 2 features ----
sub = np.sort(np.random.default_rng(SEED).choice(n, 150, replace=False))
S2 = StandardScaler().fit_transform(df.loc[sub, [a, b]]); p2 = PCA(random_state=SEED).fit(S2); Z2 = p2.transform(S2)
assert np.allclose(Z2, (S2 - S2.mean(0)) @ p2.components_.T)      # Z = Xc V consistency check
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.4))
ax1.scatter(S2[:, 0], S2[:, 1], s=22, color="#6b7280", alpha=.7)
for i, (v, l, c) in enumerate(zip(p2.components_, p2.explained_variance_, ["#d62728", "#2ca02c"])):
    ax1.annotate("", xy=v * np.sqrt(l) * 2, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=c, lw=2.5))
    ax1.text(*(v * np.sqrt(l) * 2.15), f"PC{i+1}", color=c, fontweight="bold")
ax1.set(xlabel=f"Standardised {a}", ylabel=f"Standardised {b}", title="Before: original (standardised) axes", aspect="equal")
ax2.scatter(Z2[:, 0], Z2[:, 1], s=22, color="#6b7280", alpha=.7); ax2.axhline(0, color="#2ca02c", lw=1.5); ax2.axvline(0, color="#d62728", lw=1.5)
ax2.set(xlabel=f"PC1 score ({p2.explained_variance_ratio_[0]*100:.1f}% of variance)", ylabel=f"PC2 score ({p2.explained_variance_ratio_[1]*100:.1f}%)",
        title="After: same points in PC coordinates", aspect="equal")
fig.suptitle("PCA rotates the axes (150 sampled observations, 2 features, seed 42; arrows = eigenvector × 2·sqrt(eigenvalue))", fontweight="bold", y=1.02)
save(fig, "pca-before-after-projection.png")
R["g3"] = {"evr": p2.explained_variance_ratio_.tolist(), "components": p2.components_.tolist(), "subset_n": 150}

# ---- Graph 4: scree ----
fig, ax = plt.subplots(figsize=(9.5, 5.2)); x = np.arange(1, p + 1)
ax.bar(x, evr, color="#1f6f8b"); ax.plot(x, evr, "o-", color="#e36414", ms=4)
for i in range(3): ax.annotate(f"{evr[i]*100:.1f}%", (x[i], evr[i]), xytext=(6, 4), textcoords="offset points", fontweight="bold")
ax.set(xlabel="Principal component number", ylabel="Explained variance ratio", title="Scree plot: variance explained by each component", xticks=x[::2])
save(fig, "pca-explained-variance-scree-plot.png")

# ---- Graph 5: cumulative ----
fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.plot(x, cum, "o-", color="#1f6f8b", ms=4)
for t, c in [(.90, "#e36414"), (.95, "#d62728")]:
    k = int(np.argmax(cum >= t) + 1); ax.axhline(t, color=c, ls="--", lw=1.2); ax.axvline(k, color=c, ls=":", lw=1.2)
    ax.annotate(f"{int(t*100)}% reached at {k} components ({cum[k-1]*100:.1f}%)", (k, t), xytext=(k + 1, t - .09), color=c, fontweight="bold")
ax.set(xlabel="Number of components retained", ylabel="Cumulative explained variance ratio", title="Cumulative explained variance", ylim=(0, 1.03), xticks=x[::2])
save(fig, "pca-cumulative-explained-variance.png")

# ---- Graph 6: PC1 vs PC2 ----
fig, ax = plt.subplots(figsize=(8.5, 6))
for c, lab in [("B", "Benign (B)"), ("M", "Malignant (M)")]:
    m = y == c; ax.scatter(Z[m, 0], Z[m, 1], s=26, alpha=.7, color=COL[c], label=lab, edgecolor="white", lw=.4)
ax.set(xlabel=f"PC1 ({evr[0]*100:.1f}% of variance)", ylabel=f"PC2 ({evr[1]*100:.1f}% of variance)", title="Samples in the first two principal components")
ax.legend(title="Diagnosis (colour only; PCA never saw it)")
ax.text(.02, .02, f"PC1 + PC2 together: {cum[1]*100:.1f}% of total variance", transform=ax.transAxes, bbox=dict(fc="#f1f5f9", ec="#cbd5e1", boxstyle="round"))
save(fig, "pca-two-dimensional-scatter.png")
# diagnosis separation along PC1 (descriptive only)
R["g6"] = {"pc1_mean_B": float(Z[y == "B", 0].mean()), "pc1_mean_M": float(Z[y == "M", 0].mean())}

# ---- Graph 7: loadings (sklearn convention: eigenvector * sqrt(eigenvalue)); first 5 PCs, all 30 features ----
load = pca.components_.T * np.sqrt(lam); NPC = 5
L = pd.DataFrame(load[:, :NPC], index=FEATURES, columns=[f"PC{i+1}" for i in range(NPC)])
fig, ax = plt.subplots(figsize=(7.5, 11))
sns.heatmap(L, annot=True, fmt=".2f", annot_kws={"size": 7.5}, cmap="coolwarm", center=0, vmin=-1, vmax=1, cbar_kws={"label": "Loading (corr. of feature with PC score)"}, ax=ax)
ax.set_title("Loadings of the 30 features on the first 5 PCs\n(subset = 5 largest-variance components)")
save(fig, "pca-component-loadings-heatmap.png")
top = L["PC1"].abs().sort_values(ascending=False).head(3)
R["g7"] = {"pc1_top": {k: float(L.loc[k, "PC1"]) for k in top.index}, "pc1_min_abs": float(L["PC1"].abs().min()), "pc1_n_same_sign": int((np.sign(L["PC1"]) == np.sign(L["PC1"].iloc[0])).sum())}

# ---- Graph 8: reconstruction MSE (mean over all n*p entries, standardised units) ----
ks = list(range(1, p + 1)); mse = []
for k in ks:
    pk = PCA(n_components=k, random_state=SEED).fit(Xs); mse.append(float(np.mean((Xs - pk.inverse_transform(pk.transform(Xs))) ** 2)))
fig, ax = plt.subplots(figsize=(9.5, 5.2)); ax.plot(ks, mse, "o-", color="#6c4f9e", ms=4)
for k in [1, 2, 5, R["k95"], 10]:
    ax.annotate(f"k={k}: {mse[k-1]:.4f}", (k, mse[k-1]), xytext=(8, 8), textcoords="offset points", fontsize=9)
ax.set(xlabel="Number of components retained (k)", ylabel="Mean squared reconstruction error (standardised units)", title="Reconstruction error vs components retained", xticks=ks[::2])
save(fig, "pca-reconstruction-error.png")
R["mse"] = {str(k): m for k, m in zip(ks, mse)}
R["mse_check_theory_k2"] = float(lam[2:].sum() * (n - 1) / n / p)   # discarded variance per entry (matches mse[1])

# ---- Scaled vs unscaled PCA (Chapter 6) ----
raw = df[FEATURES].to_numpy(float); pu = PCA(random_state=SEED).fit(raw)
R["unscaled"] = {"pc1": float(pu.explained_variance_ratio_[0]), "pc2": float(pu.explained_variance_ratio_[1]),
                 "top_loading_feature": FEATURES[int(np.argmax(np.abs(pu.components_[0])))], "top_loading": float(np.abs(pu.components_[0]).max()),
                 "max_var_feature": df[FEATURES].var().idxmax(), "max_var": float(df[FEATURES].var().max()), "min_var": float(df[FEATURES].var().min())}

# ---- Leakage-safe predictive comparison (supervised; diagnosis used ONLY here) ----
Xtr, Xte, ytr, yte = train_test_split(df[FEATURES], (y == "M").astype(int), test_size=.25, stratify=y, random_state=SEED)
base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Xtr, ytr)
pcam = make_pipeline(StandardScaler(), PCA(n_components=0.95, random_state=SEED), LogisticRegression(max_iter=5000)).fit(Xtr, ytr)
pc2m = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=SEED), LogisticRegression(max_iter=5000)).fit(Xtr, ytr)
ev = lambda m: {"accuracy": float(accuracy_score(yte, m.predict(Xte))), "auc": float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))}
R["model"] = {"split": "75/25 stratified, seed 42", "baseline_30": ev(base), "pca_95": ev(pcam), "pca_95_k": int(pcam[1].n_components_), "pca_2": ev(pc2m)}
Path("results.json").write_text(json.dumps(R, indent=2))
print(json.dumps({k: R[k] for k in ["k90", "k95", "k99", "corr", "g7", "g6", "unscaled", "model", "mse_check_theory_k2"]}, indent=1))
print("EVR first 5:", np.round(evr[:5], 4), "| mse k=2:", round(mse[1], 5), "| g3:", R["g3"]["evr"])
