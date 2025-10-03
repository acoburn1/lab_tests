import os, glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from Eval.PCA import get_pcns_mod_lat   # import directly

def plot_generalization_vs_dimensionality_diff(
    data_dir,
    epoch,
    num_features=11,
    hidden=True,
    save_dir="Results/Analysis/Plots/scatter_dimensionality",
    show_plots=False,
    include_e0=False
):
    """
    Scatter of generalization (%Mod response) vs Mod–Lat principal dimensionality (k95_mod − k95_lat).
    Each dot = one model at the given epoch.

    X-axis: (k95_mod - k95_lat) from PCA on hidden activations
    Y-axis: %Mod preference from 3:3 ratio tests
    """

    os.makedirs(save_dir, exist_ok=True)
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not npz_files:
        print(f"No NPZ files found in {data_dir}")
        return

    test_data_key = 'hidden_ratio_tests' if hidden else 'output_ratio_tests'
    activation_key = 'hidden_activations'   # assumed key

    kdiff_per_model = []
    mod_pref_per_model = []

    for f in npz_files:
        d = np.load(f, allow_pickle=True)

        # --- X values: k95_mod - k95_lat ---
        if activation_key not in d or epoch >= len(d[activation_key]):
            continue
        A = np.asarray(d[activation_key][epoch], float)
        if A.shape[0] != 2 * num_features:
            continue
        km, kl = get_pcns_mod_lat(A, num_features)
        kdiff = km - kl

        # --- Y values: generalization from ratio tests ---
        if test_data_key not in d or epoch >= len(d[test_data_key]):
            continue
        test_epoch = d[test_data_key][epoch]
        if '3:3' not in test_epoch:
            continue

        ratio_data = test_epoch['3:3']
        # average %mod preference across all sets
        rates = []
        for set_name, set_data in ratio_data.items():
            mod_corrs = np.asarray(set_data["mod"], float)
            lat_corrs = np.asarray(set_data["lat"], float)
            avg_mod = np.mean(mod_corrs, axis=1)
            avg_lat = np.mean(lat_corrs, axis=1)
            mod_pref = np.mean(avg_mod > avg_lat)
            rates.append(mod_pref)
        if not rates:
            continue
        mod_pref_rate = np.mean(rates)

        kdiff_per_model.append(kdiff)
        mod_pref_per_model.append(mod_pref_rate)

    if not kdiff_per_model:
        print("No valid data at this epoch.")
        return

    # --- Plot ---
    layer = 'Hidden' if hidden else 'Output'
    epoch_display = epoch if include_e0 else epoch + 1
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(kdiff_per_model, mod_pref_per_model, s=60, alpha=0.7, zorder=3)

    if len(kdiff_per_model) >= 2:
        r, pval = pearsonr(kdiff_per_model, mod_pref_per_model)
        z = np.polyfit(kdiff_per_model, mod_pref_per_model, 1)
        p = np.poly1d(z)
        xs = np.linspace(min(kdiff_per_model), max(kdiff_per_model), 100)
        ax.plot(xs, p(xs), color='red', linewidth=2, alpha=0.8,
                label=f'r = {r:.3f}, p = {pval:.3f}')

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1, label='Chance (50%)')
    ax.set_xlabel("Dimensionality Difference (k95_mod - k95_lat)", fontsize=12)
    ax.set_ylabel("Generalization (%Mod preference)", fontsize=12)
    ax.set_title(f"Generalization vs Dimensionality - {layer} (Epoch {epoch_display})", fontsize=14)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"scatter_dimensionality_kdiff_{layer[0].lower()}e{epoch_display}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close()

    return {
        "epoch": epoch,
        "kdiff": kdiff_per_model,
        "generalization": mod_pref_per_model,
        "save_path": out_path
    }
