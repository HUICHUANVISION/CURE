import matplotlib.pyplot as plt
import numpy as np

labels = ["DA removed", "CS removed", "WS removed"]
auc_delta = [-0.010, -0.004, -0.003]
f1_delta  = [-0.010, -0.014, -0.002]
mcc_delta = [-0.020, -0.017, -0.006]

x = np.arange(len(labels))
width = 0.25

plt.figure(figsize=(6,4))
plt.bar(x - width, auc_delta, width, label='AUC')
plt.bar(x, f1_delta, width, label='F1')
plt.bar(x + width, mcc_delta, width, label='MCC')
plt.axhline(0, color='black', linewidth=0.8)
plt.ylabel("Δ (Removed − Full)")
plt.title("Component Contribution in CURE")
plt.xticks(x, labels, rotation=15)
plt.legend()
plt.tight_layout()
plt.savefig("runs/rq3_ablation_bar.pdf")