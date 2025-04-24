import glob
import json
import pandas as pd
import matplotlib.pyplot as plt

# BASE_DIR = "./on_GPU"

# training diagnostics
histories = []
for BASE_DIR in ['./on_CPU', './on_GPU']:
	for csv in glob.glob(f"{BASE_DIR}/analytics/history_*.csv"):
		df = pd.read_csv(csv)                       # epochs x metrics
		tag = "_".join(csv.split("_")[1:4])         # like sgd_none_42.csv
		df['condition'] = tag.replace(".csv", "")
		histories.append(df)
    
logs = pd.concat(histories, ignore_index=True)

# average across seeds, optimiser, augmentation for clarity
mean_log = logs.groupby('epoch').mean(numeric_only=True)

plt.figure()
plt.plot(mean_log.index, mean_log['train_acc'],
         color='#228B22', label='train')
plt.plot(mean_log.index, mean_log['test_acc'],
         color='#800080', label='validation')
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.grid(True)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"train_val_accuracy.png")

plt.figure()
plt.plot(mean_log.index, mean_log['train_loss'],
         color='#008080', label='train')  # teal
plt.plot(mean_log.index, mean_log['test_loss'],
         color='#FF00FF', label='validation')  # magenta
plt.xlabel("epoch")
plt.ylabel("cross‑entropy loss")
plt.legend()
plt.grid(True)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"train_val_loss.png")

# robustness curves
with open(f"./combined_results.json") as f:
    res = json.load(f)

df = pd.DataFrame(res)
records = []  # one row per sigma

for _, row in df.iterrows():
    for sigma, acc in row['robustness'].items():
        records.append({
            'optimizer': row['optimizer'],
            'augmentation': row['augmentation'],
            'sigma': float(sigma),
            'acc': acc,
        })

rob_df = pd.DataFrame(records)
pivot = rob_df.groupby(['optimizer', 'sigma']).acc.mean().unstack(0)

ax = pivot.plot(marker='o', linestyle='-', color=['#FF0000', '#00008B'])
plt.xlabel("Gaussian noise sigma")
plt.ylabel("accuracy")
plt.title("Noise robustness")
plt.grid(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"./robustness_curve.png")
print("saved robustness_curve.png")
