import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# -------------------------------------
# Load prediction dictionaries safely
# -------------------------------------
pred_lstm_dict = np.load("pred_lstm.npy", allow_pickle=True).item()
pred_gru_dict = np.load("pred_gru.npy", allow_pickle=True).item()
pred_att_lstm_dict = np.load("pred_att_lstm.npy", allow_pickle=True).item()

# -------------------------------------
# Create plots directory
# -------------------------------------
os.makedirs("plotsss", exist_ok=True)

# -------------------------------------
# Combine predictions from all companies
# -------------------------------------
def combine_all(pred_dict):
    actual_all = []
    pred_all = []
    for company, (y_true, y_pred) in pred_dict.items():
        actual_all.extend(y_true)
        pred_all.extend(y_pred)
    return np.array(actual_all), np.array(pred_all)

actual_lstm, pred_lstm = combine_all(pred_lstm_dict)
actual_gru, pred_gru = combine_all(pred_gru_dict)
actual_att, pred_att_lstm = combine_all(pred_att_lstm_dict)

# -------------------------------------
# Align lengths
# -------------------------------------
min_len = min(len(actual_lstm), len(pred_lstm), len(pred_gru), len(pred_att_lstm))
actual = actual_lstm[:min_len]
pred_lstm = pred_lstm[:min_len]
pred_gru = pred_gru[:min_len]
pred_att_lstm = pred_att_lstm[:min_len]

# -------------------------------------
# Plot combined graph and save
# -------------------------------------
plt.figure(figsize=(14, 6))
plt.plot(actual, label="Actual", color="black", linewidth=2)
plt.plot(pred_lstm, label="LSTM", linestyle="--", color="blue")
plt.plot(pred_att_lstm, label="Attention LSTM", linestyle="--", color="red")
plt.plot(pred_gru, label="Fine Tuned LSTM-GRU", linestyle="--", color="green")

plt.title("Actual vs Predicted Stock Prices (All Companies Combined)", fontsize=14)
plt.xlabel("Time Steps", fontsize=12)
plt.ylabel("Stock Price", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

plot_path = os.path.join("plotsss", "combined_predictions.png")
plt.savefig(plot_path)
plt.close()
print(f"✅ Combined plot saved to {plot_path}")

# -------------------------------------
# Evaluate and save only average metrics per model
# -------------------------------------
eval_path = os.path.join("plotsss", "model_evaluation.txt")

def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

# Collect metrics per model across all companies
metrics_per_model = {"LSTM": [], "GRU": [], "Attention LSTM": []}

model_list = [
    ("LSTM", pred_lstm_dict),
    ("GRU", pred_gru_dict),
    ("Attention LSTM", pred_att_lstm_dict)
]

for model_name, pred_dict in model_list:
    for company in pred_dict.keys():
        y_true, y_pred = pred_dict[company]
        metrics_per_model[model_name].append(compute_metrics(y_true, y_pred))

# Write only average metrics to file
with open(eval_path, "w") as f:
    f.write("AVERAGE METRICS PER MODEL ACROSS ALL COMPANIES\n")
    f.write("=" * 60 + "\n")
    f.write(f"{'Model':<20}{'MSE':>10}{'RMSE':>10}{'MAE':>10}{'R²':>10}\n")
    
    for model_name, values in metrics_per_model.items():
        mse_avg = np.mean([v[0] for v in values])
        rmse_avg = np.mean([v[1] for v in values])
        mae_avg = np.mean([v[2] for v in values])
        r2_avg = np.mean([v[3] for v in values])
        f.write(f"{model_name:<20}{mse_avg:>10.4f}{rmse_avg:>10.4f}{mae_avg:>10.4f}{r2_avg:>10.4f}\n")

print(f"✅ Average model evaluation metrics saved to {eval_path}")
