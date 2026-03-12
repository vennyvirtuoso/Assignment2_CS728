import re
import matplotlib.pyplot as plt
import os

def plot_all_thresholds(log_file_path):
    if not os.path.exists("plots"):
        os.makedirs("plots")
    
    # We want the output name to match the final state naming convention for your report
    base_name = os.path.basename(log_file_path).replace(".log", "")
    
    data_points = []

    # Regex to capture iterations and all three specific thresholds
    pattern = re.compile(
        r"Iter (?P<iter>\d+):.*err@0.05 (?P<err05>[\d\.]+)%, err@0.10 (?P<err10>[\d\.]+)%, err@0.20 (?P<err20>[\d\.]+)%"
    )

    if not os.path.exists(log_file_path):
        print(f"File not found: {log_file_path}")
        return

    with open(log_file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                # Store as a tuple: (iteration, err05, err10, err20)
                data_points.append((
                    int(match.group("iter")),
                    float(match.group("err05")),
                    float(match.group("err10")),
                    float(match.group("err20"))
                ))

    if not data_points:
        print(f"No threshold data found in {log_file_path}")
        return

    # 1. Sort based on the original iteration count to ensure chronological order
    data_points.sort(key=lambda x: x[0])

    # 2. Extract values and create sequential steps (1, 2, 3...)
    steps = range(1, len(data_points) + 1)
    err_05 = [x[1] for x in data_points]
    err_10 = [x[2] for x in data_points]
    err_20 = [x[3] for x in data_points]

    plt.figure(figsize=(10, 6))
    
    # 3. Plot using the sequential 'steps'
    plt.plot(steps, err_05, label='Error @ 0.05 (Tight)', color='#e74c3c', linewidth=2)
    plt.plot(steps, err_10, label='Error @ 0.10 (Mid)', color='#3498db', linewidth=2)
    plt.plot(steps, err_20, label='Error @ 0.20 (Loose)', color='#2ecc71', linewidth=2)
    
    # 4. Heading and Labels
    plt.xlabel('Step (Checkpoint Count)')
    plt.ylabel('Error Rate (%)')
    plt.title('MAE/error@thresholds') # Explicit requested title
    
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 105) 
    
    # Saving with the final_state_err suffix for compatibility
    output_path = f"plots/{base_name}_final_state_err.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

# Run for your specific Task 2 logs
log_files = [
    "logs/B1_mul_rnn_tanh_noclip.log",
    "logs/B2_mul_gru_noclip.log"
]

for log in log_files:
    plot_all_thresholds(log)