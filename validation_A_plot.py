import re
import matplotlib.pyplot as plt
import os

def plot_memorization_metrics(log_file_path):
    if not os.path.exists("plots"):
        os.makedirs("plots")
    
    base_name = os.path.basename(log_file_path).replace(".log", "")
    data_points = []

    pattern = re.compile(
        r"Iter (?P<iter>\d+):.*valid error (?P<err>[\d\.]+)%,.*valid nll (?P<nll>[\d\.]+)"
    )

    if not os.path.exists(log_file_path):
        print(f"File not found: {log_file_path}")
        return

    with open(log_file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                data_points.append({
                    "iter": int(match.group("iter")),
                    "err": float(match.group("err")),
                    "nll": float(match.group("nll"))
                })

    if not data_points:
        print(f"No memorization data found in {log_file_path}")
        return

    data_points.sort(key=lambda x: x["iter"])

    steps = range(1, len(data_points) + 1)
    valid_errors = [x["err"] for x in data_points]
    valid_nlls = [x["nll"] for x in data_points]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_err = '#2c3e50'
    ax1.set_xlabel('Step (Checkpoint Count)')
    ax1.set_ylabel('Global Valid Error (%)', color=color_err, fontweight='bold')
    ax1.plot(steps, valid_errors, color=color_err, linewidth=2.5, label='Valid Error')
    ax1.tick_params(axis='y', labelcolor=color_err)
    ax1.set_ylim(-5, 105)

    ax2 = ax1.twinx()
    color_nll = '#e67e22'
    ax2.set_ylabel('Valid NLL', color=color_nll, fontweight='bold')
    ax2.plot(steps, valid_nlls, color=color_nll, linewidth=2, linestyle='--', label='Valid NLL')
    ax2.tick_params(axis='y', labelcolor=color_nll)

    plt.title(f'Learning Progress: Error vs NLL - {base_name}')
    fig.tight_layout()
    plt.grid(True, linestyle=':', alpha=0.6)

    output_path = f"plots/{base_name}_final_state_err.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

log_files = [
    "logs/A1_mem_rnn_tanh_noclip.log",
    "logs/A2_mem_rnn_tanh_clip005.log",
    "logs/A3_mem_rnn_tanh_clip001.log",
    "logs/A4_mem_gru_noclip.log",
    "logs/A5_mem_gru_clip005.log"
]

for log in log_files:
    plot_memorization_metrics(log)