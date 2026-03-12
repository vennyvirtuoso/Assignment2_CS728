import numpy as np
import matplotlib.pyplot as plt
import glob
import os

if not os.path.exists("plots"):
    os.makedirs("plots")

for file_path in glob.glob("*final_state.npz"):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    z = np.load(file_path)
    
    # 1. Gradient Norms
    g = z["grad_time"][-1]
    g = g[np.isfinite(g)]
    plt.figure()
    plt.hist(np.log10(g + 1e-12), bins=60)
    plt.title(f"log10 ||dL/dh_t|| - {base_name}")
    plt.savefig(f"plots/{base_name}_grad.png")
    plt.close()

    # 2. Hidden Saturation
    s = z["sat_time"][-1]
    s = s[np.isfinite(s)]
    plt.figure()
    plt.hist(s, bins=60, range=(0,1))
    plt.title(f"Saturation - {base_name}")
    plt.savefig(f"plots/{base_name}_sat.png")
    plt.close()

    # 3. Validation Error
    plt.figure()
    plt.plot(z["valid_error"])
    plt.title(f"Val Error - {base_name}")
    plt.savefig(f"plots/{base_name}_err.png")
    plt.close()

    if "rho_Whh" in z:
            rho = z["rho_Whh"]
            plt.figure()
            plt.plot(rho, marker='o', markersize=4)
            plt.axhline(y=1.0, color='r', linestyle='--', label='Stability Threshold (1.0)')
            plt.title(f"Spectral Radius (rho) - {base_name}")
            plt.xlabel("Checkpoints")
            plt.ylabel("rho(Whh)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f"plots/{base_name}_rho.png")
            plt.close()

    # 4. GRU Gates (if enabled)
    if "gru" in base_name.lower():
        for gate in ["z", "r"]:
            data = z[f"gate_{gate}_sat_time"][-1]
            data = data[np.isfinite(data)]
            plt.figure()
            plt.hist(data, bins=60, range=(0,1))
            plt.title(f"Gate {gate.upper()} Saturation - {base_name}")
            plt.savefig(f"plots/{base_name}_gate_{gate}.png")
            plt.close()