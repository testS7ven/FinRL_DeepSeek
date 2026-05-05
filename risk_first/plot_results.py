import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_equity_curves(results_dir="risk_first/results"):
    res_path = Path(results_dir)
    if not res_path.exists():
        print(f"Dossier {results_dir} introuvable.")
        return

    # Charger le benchmark (l'indice NASDAQ)
    bench_file = res_path / "A_PPO_baseline_benchmark.npy"
    if bench_file.exists():
        benchmark = np.load(bench_file)
    else:
        benchmark = None

    # Les 6 configurations de l'Ablation Study
    configs = [
        "A_PPO_baseline", 
        "B_PPO_LLM", 
        "C_CPPO_baseline", 
        "D_CPPO_confidence", 
        "E_CPPO_shaping", 
        "F_full_model"
    ]
    # Couleurs académiques pour bien différencier les courbes
    colors = ["#95a5a6", "#34495e", "#3498db", "#e67e22", "#9b59b6", "#2ecc71"]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 6))
    
    if benchmark is not None:
        plt.plot(benchmark, label="Benchmark (NASDAQ)", color="#e74c3c", linestyle="--", linewidth=2, alpha=0.8)

    # Tracer chaque courbe qui a terminé son entraînement
    for cfg, color in zip(configs, colors):
        eq_file = res_path / f"{cfg}_equity.npy"
        if eq_file.exists():
            equity = np.load(eq_file)
            # Simplifier le nom pour la légende du papier
            label_name = cfg.replace("_", " ")
            plt.plot(equity, label=label_name, color=color, linewidth=2)
            
    plt.title("Ablation Study: Portfolio Equity Curves over Time", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Trading Days", fontsize=12)
    plt.ylabel("Portfolio Value ($)", fontsize=12)
    plt.legend(loc="upper left", frameon=True, shadow=True)
    plt.tight_layout()
    
    # Sauvegarde en haute qualité pour NeurIPS
    out_path = res_path / "ablation_equity_curve.png"
    plt.savefig(out_path, dpi=300)
    print(f"Graphique généré avec succès : {out_path}")

if __name__ == "__main__":
    plot_equity_curves()
