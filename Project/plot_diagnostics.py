import glob
import json
import pandas as pd
import matplotlib.pyplot as plt

# training diagnostics
histories = []
for csv in glob.glob("analytics/history_*.csv"):
    df = pd.read_csv(csv)                       # epochs × metrics
    tag = "_".join(csv.split("_")[1:4])         # like sgd_none_42.csv
    df['condition'] = tag.replace(".csv", "")
    histories.append(df)
logs = pd.concat(histories, ignore_index=True)

# average across seeds, optimiser, augmentation for clarity
mean_log = logs.groupby('epoch').mean(numeric_only=True)

plt.figure()
plt.plot(mean_log.index, mean_log['train_acc'], label='train')
plt.plot(mean_log.index, mean_log['test_acc'], label='validation')
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("train_val_accuracy.png")

plt.figure()
plt.plot(mean_log.index, mean_log['train_loss'], label='train')
plt.plot(mean_log.index, mean_log['test_loss'], label='validation')
plt.xlabel("epoch")
plt.ylabel("cross‑entropy loss")
plt.legend()
plt.tight_layout()
plt.savefig("train_val_loss.png")

# robustness curves
with open("results.json") as f:
    res = json.load(f)

df = pd.DataFrame(res)
records = [] # one row per sigma

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

# sigma on x‑axis, one line per optimiser
pivot.plot(marker='o')
plt.xlabel("Gaussian noise sigma")
plt.ylabel("accuracy")
plt.title("Noise robustness")
plt.tight_layout()
plt.savefig("robustness_curve.png")
print("saved robustness_curve.png")
