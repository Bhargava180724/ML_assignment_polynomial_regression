"""
generate_plots.py -- Report-quality plots for ML Assignment 1 (BT2024166)
Part 1 (var1): 5 plots  |  Part 2 (var2): 5 plots
Saved to: plots\\

Reads the CV results produced by train_var1.py / train_var2.py
(metrics_part1.csv / metrics_part2.csv) and re-applies the exact same
"simplest model within 1% MSE tolerance" selection rule used in those
scripts to find the selected model/degree -- nothing here is hardcoded,
everything is computed from the data so the plots always match whatever
train_var1.py / train_var2.py most recently produced.
"""
import sys, io, os, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline

# ──────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────
TRAIN_V1  = "BT2024166_train_var1.csv"
TRAIN_V2  = "BT2024166_train_var2.csv"
CSV1_PATH = "metrics_part1.csv"
CSV2_PATH = "metrics_part2.csv"
OUT       = "plots"
os.makedirs(OUT, exist_ok=True)

MAX_DEG_1 = 10
MAX_DEG_2 = 20
TOLERANCE = 0.01   # same 1% relative tolerance used in train_var1.py / train_var2.py

# ──────────────────────────────────────────────────────────
# Global style
# ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 11,
    'axes.titlesize'   : 13,
    'axes.titleweight' : 'bold',
    'axes.labelsize'   : 12,
    'legend.fontsize'  : 8.5,
    'xtick.labelsize'  : 10,
    'ytick.labelsize'  : 10,
    'savefig.dpi'      : 300,
    'savefig.bbox'     : 'tight',
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'grid.linestyle'   : '--',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.facecolor'   : '#FAFAFA',
    'figure.facecolor' : 'white',
})

# ──────────────────────────────────────────────────────────
# Colour / style per model (same 10 candidates as train_var1.py / train_var2.py)
# ──────────────────────────────────────────────────────────
MLABELS = [
    'OLS Linear',
    'Ridge(a=0.1)', 'Ridge(a=1.0)', 'Ridge(a=10)',
    'Lasso(a=0.001)', 'Lasso(a=0.01)', 'Lasso(a=0.1)',
    'EN(a=0.001)', 'EN(a=0.01,r=0.5)', 'EN(a=0.01,r=0.7)',
]
MCOLORS = [
    '#c0392b',
    '#5dade2', '#2e86c1', '#1a5276',
    '#f9e79f', '#f39c12', '#d35400',
    '#a9dfbf', '#27ae60', '#1e8449',
]
LW = [1.8, 1.4, 2.0, 1.4,  1.4, 2.0, 1.4,  1.4, 1.4, 1.4]
LS = ['-', '--', '-', ':',  '--', '-', ':',  '--', '-', ':']

MODEL_NAME_MAP = {
    'Linear Regression':            'OLS Linear',
    'Ridge (alpha=0.1)':             'Ridge(a=0.1)',
    'Ridge (alpha=1.0)':             'Ridge(a=1.0)',
    'Ridge (alpha=10.0)':            'Ridge(a=10)',
    'Lasso (alpha=0.001)':           'Lasso(a=0.001)',
    'Lasso (alpha=0.01)':            'Lasso(a=0.01)',
    'Lasso (alpha=0.1)':             'Lasso(a=0.1)',
    'ElasticNet (a=0.001, l1=0.5)':  'EN(a=0.001)',
    'ElasticNet (a=0.01, l1=0.5)':   'EN(a=0.01,r=0.5)',
    'ElasticNet (a=0.01, l1=0.7)':   'EN(a=0.01,r=0.7)',
}
COMPLEXITY_RANK = {name: i for i, name in enumerate(MODEL_NAME_MAP)}
LABEL_TO_IDX = {label: idx for idx, label in enumerate(MLABELS)}

FAMILIES = {'OLS': '#c0392b', 'Ridge': '#2e86c1', 'Lasso': '#f39c12', 'ElasticNet': '#27ae60'}

def family_of(raw_name):
    if raw_name.startswith('Linear'):
        return 'OLS'
    if raw_name.startswith('Ridge'):
        return 'Ridge'
    if raw_name.startswith('Lasso'):
        return 'Lasso'
    return 'ElasticNet'

def build_model(raw_name):
    """Re-create the exact estimator instance for a given candidate name
    (mirrors get_candidate_models() in train_var1.py / train_var2.py)."""
    if raw_name == 'Linear Regression':
        return LinearRegression()
    if raw_name.startswith('Ridge'):
        alpha = float(raw_name.split('=')[1].rstrip(')'))
        return Ridge(alpha=alpha)
    if raw_name.startswith('Lasso'):
        alpha = float(raw_name.split('=')[1].rstrip(')'))
        return Lasso(alpha=alpha, max_iter=10000, tol=0.01, random_state=42)
    # ElasticNet (a=X, l1=Y)
    parts = raw_name.replace('ElasticNet (', '').rstrip(')').split(', ')
    alpha = float(parts[0].split('=')[1])
    l1 = float(parts[1].split('=')[1])
    return ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=10000, tol=0.01, random_state=42)

def build_matrices_from_csv(df, max_deg):
    mse_mat = np.full((max_deg, len(MLABELS)), np.nan)
    r2_mat  = np.full((max_deg, len(MLABELS)), np.nan)
    for _, row in df.iterrows():
        deg = int(row['degree'])
        if deg > max_deg:
            continue
        label = MODEL_NAME_MAP.get(row['model_name'], row['model_name'])
        m_idx = LABEL_TO_IDX[label]
        mse_mat[deg - 1, m_idx] = float(row['cv_mse'])
        r2_mat[deg - 1, m_idx]  = float(row['cv_r2'])
    return mse_mat, r2_mat

def select_model(df):
    """Exactly replicates the selection logic in train_var1.py / train_var2.py:
    simplest (lowest degree, then lowest complexity_rank) model within 1% of the
    global best CV MSE."""
    df = df.copy()
    df['complexity_rank'] = df['model_name'].map(COMPLEXITY_RANK)
    best_row  = df.loc[df['cv_mse'].idxmin()]
    threshold = best_row['cv_mse'] * (1 + TOLERANCE)
    within    = df[df['cv_mse'] <= threshold].sort_values(['degree', 'complexity_rank'])
    sel       = within.iloc[0]
    return best_row, sel

def legend_handles(indices=range(10)):
    return [Line2D([0], [0], color=MCOLORS[i], lw=LW[i],
                   linestyle=LS[i], label=MLABELS[i]) for i in indices]

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  Saved:', name)

def build_pipe(model, deg, X, y):
    pipe = Pipeline([
        ('poly',  PolynomialFeatures(degree=deg, include_bias=True)),
        ('scale', StandardScaler()),
        ('model', model),
    ])
    pipe.fit(X, y)
    return pipe

def make_part_plots(part_label, train_path, feats, csv_path, max_deg, out_prefix):
    print(f'\n--- {part_label} ---')
    cv_df = pd.read_csv(csv_path)
    MSE, R2 = build_matrices_from_csv(cv_df, max_deg)
    D = np.arange(1, max_deg + 1)

    best_row, sel_row = select_model(cv_df)
    sel_name, sel_deg   = sel_row['model_name'], int(sel_row['degree'])
    sel_mse, sel_r2      = sel_row['cv_mse'], sel_row['cv_r2']
    raw_name, raw_deg    = best_row['model_name'], int(best_row['degree'])
    raw_mse               = best_row['cv_mse']
    sel_label = MODEL_NAME_MAP.get(sel_name, sel_name)
    sel_idx   = LABEL_TO_IDX[sel_label]
    print(f'  Raw best : {raw_name}, degree={raw_deg}, CV MSE={raw_mse:.4f}')
    print(f'  Selected : {sel_name}, degree={sel_deg}, CV MSE={sel_mse:.4f} (1% tolerance rule)')

    train_df = pd.read_csv(train_path)
    X = train_df[feats].values
    y = train_df['y'].values
    pipe = build_pipe(build_model(sel_name), sel_deg, X, y)
    yh   = pipe.predict(X)
    res  = y - yh
    r2_train  = 1 - np.var(res) / np.var(y)
    mse_train = np.mean(res ** 2)
    print(f'  Final model fitted. Train R2={r2_train:.4f}, Train MSE={mse_train:.4f}')

    # ---- Plot 1: CV MSE vs Degree (log scale), all 10 candidates ----
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i in range(10):
        ax.semilogy(D, MSE[:, i], color=MCOLORS[i], lw=LW[i], linestyle=LS[i],
                    marker='o', ms=4, label=MLABELS[i], alpha=0.9)
    ax.scatter([sel_deg], [sel_mse], s=220, zorder=9, color='#f39c12',
               edgecolors='black', linewidths=1.5, marker='*')
    ax.annotate(f'Selected model\n{sel_label}, deg={sel_deg}\nCV MSE={sel_mse:.4f}',
                xy=(sel_deg, sel_mse),
                xytext=(min(sel_deg + 2.2, max_deg - 2), sel_mse * 0.4 + 1e-6),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='#555'),
                bbox=dict(boxstyle='round,pad=0.3', fc='#fff3cd', ec='#f39c12', alpha=0.95))
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('CV MSE  (log scale)')
    ax.set_title(f'{part_label} — CV MSE vs Degree: All Models')
    ax.set_xticks(D)
    if max_deg > 10:
        ax.tick_params(axis='x', rotation=45)
    ax.legend(handles=legend_handles(), loc='upper right', ncol=2, framealpha=0.9)
    ax.set_xlim(0.5, max_deg + 0.5)
    save(fig, f'{out_prefix}_1_cv_mse_vs_degree.png')

    # ---- Plot 2: CV R2 vs Degree, all 10 candidates ----
    fig, ax = plt.subplots(figsize=(10, 5.5))
    lower_clip = max(np.nanmin(R2), -1.0)
    for i in range(10):
        ax.plot(D, np.clip(R2[:, i], lower_clip, 1.0), color=MCOLORS[i], lw=LW[i],
                linestyle=LS[i], marker='o', ms=4, label=MLABELS[i], alpha=0.9)
    ax.axhline(0, color='grey', lw=0.8, linestyle='--')
    ax.scatter([sel_deg], [sel_r2], s=220, zorder=9, color='#f39c12',
               edgecolors='black', linewidths=1.5, marker='*')
    ax.annotate(f'Selected: CV R2={sel_r2:.4f}\n{sel_label}, deg={sel_deg}',
                xy=(sel_deg, sel_r2),
                xytext=(min(sel_deg + 2, max_deg - 2), max(lower_clip + 0.1, sel_r2 - 0.25)),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='#555'),
                bbox=dict(boxstyle='round,pad=0.3', fc='#fff3cd', ec='#f39c12', alpha=0.95))
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel(f'CV R2  (clamped at {lower_clip:.2f})')
    ax.set_title(f'{part_label} — CV R2 vs Degree: All Models')
    ax.set_xticks(D)
    if max_deg > 10:
        ax.tick_params(axis='x', rotation=45)
    ax.legend(handles=legend_handles(), loc='lower right', ncol=2, framealpha=0.9)
    ax.set_xlim(0.5, max_deg + 0.5)
    save(fig, f'{out_prefix}_2_cv_r2_vs_degree.png')

    # ---- Plot 3: Best CV MSE per Degree (bar chart, colour = winning family) ----
    best_mse_per_deg = np.nanmin(MSE, axis=1)
    best_idx_per_deg = np.nanargmin(MSE, axis=1)
    fig, ax = plt.subplots(figsize=(max(9, max_deg * 0.55), 5))
    bars = ax.bar(D, best_mse_per_deg, color=[MCOLORS[i] for i in best_idx_per_deg],
                  edgecolor='white', linewidth=1.2, zorder=3)
    for bar, val, mi in zip(bars, best_mse_per_deg, best_idx_per_deg):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012 * max(best_mse_per_deg),
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.scatter([sel_deg], [best_mse_per_deg[sel_deg - 1]], s=240, zorder=10,
               color='gold', edgecolors='black', linewidths=1.5, marker='*')
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('Best CV MSE')
    ax.set_title(f'{part_label} — Best CV MSE per Degree\n(bar colour = winning model family)')
    ax.set_xticks(D)
    if max_deg > 10:
        ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, max(best_mse_per_deg) * 1.25)
    ax.legend([Patch(fc=c, label=n) for n, c in FAMILIES.items()],
               FAMILIES.keys(), loc='upper right', fontsize=9)
    save(fig, f'{out_prefix}_3_best_mse_per_degree.png')

    # ---- Plot 4: Actual vs Predicted (training set) ----
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    sc = ax.scatter(y, yh, alpha=0.4, s=18, c=np.abs(res), cmap='YlOrRd', edgecolors='none')
    plt.colorbar(sc, ax=ax, label='|Residual|')
    mn, mx = min(y.min(), yh.min()), max(y.max(), yh.max())
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1.5, label='Perfect fit')
    ax.text(0.05, 0.93, f'Train R2  = {r2_train:.4f}\nTrain MSE = {mse_train:.4f}',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', fc='white', ec='#ccc', alpha=0.92))
    ax.set_xlabel('Actual y')
    ax.set_ylabel('Predicted y')
    ax.set_title(f'{part_label} — Actual vs Predicted\n({sel_label}, Degree {sel_deg}, Training Set)')
    ax.legend(fontsize=9)
    ax.set_aspect('equal', 'box')
    save(fig, f'{out_prefix}_4_actual_vs_predicted.png')

    # ---- Plot 5: Residuals Analysis (2-panel) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.scatter(yh, res, alpha=0.35, s=18, color='#2e86c1', edgecolors='none')
    ax.axhline(0, color='red', lw=1.5, linestyle='--')
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs Fitted Values')
    ax = axes[1]
    mu, sd = res.mean(), res.std()
    ax.hist(res, bins=40, density=True, color='#f39c12', alpha=0.7, edgecolor='white')
    xc = np.linspace(res.min(), res.max(), 300)
    ax.plot(xc, stats.norm.pdf(xc, mu, sd), 'r-', lw=2, label=f'Normal(mu={mu:.3f}, sd={sd:.3f})')
    ax.set_xlabel('Residual Value')
    ax.set_ylabel('Density')
    ax.set_title('Residual Distribution')
    ax.legend(fontsize=9)
    fig.suptitle(f'{part_label} — Residuals Analysis  ({sel_label}, Degree {sel_deg})',
                 fontsize=12, fontweight='bold', y=1.02)
    save(fig, f'{out_prefix}_5_residuals_analysis.png')

# ══════════════════════════════════════════════════════════
#  PART 1 (var1)  —  5 PLOTS
# ══════════════════════════════════════════════════════════
make_part_plots('Part 1 (var1)', TRAIN_V1, ['x1', 'x2', 'x3', 'x4', 'x5', 'x6'],
                 CSV1_PATH, MAX_DEG_1, 'p1')

# ══════════════════════════════════════════════════════════
#  PART 2 (var2)  —  5 PLOTS
# ══════════════════════════════════════════════════════════
make_part_plots('Part 2 (var2)', TRAIN_V2, ['x1', 'x2', 'x3'],
                 CSV2_PATH, MAX_DEG_2, 'p2')

print(f'\nAll 10 plots saved to: {OUT}')
