import pandas as pd
import matplotlib.pyplot as plt

# Đọc file CSV
df_stamp = pd.read_csv('STAMP_10.csv')
df_fedavg = pd.read_csv('FedAvg_10.csv')

# Lấy Step và số lượng cosine > 0
step = df_stamp.iloc[:, 0]
stamp_pos = df_stamp.iloc[:, 3]  # Cosine > 0 cho STAMP
fedavg_pos = df_fedavg.iloc[:300, 3]  # Cosine > 0 cho FedAvg

# Tính phần trăm (giả sử có 9 client để so sánh)
stamp_percent = (1-stamp_pos / 9) * 100
fedavg_percent = (1-fedavg_pos / 9) * 100

# Áp dụng EMA smoothing (giống WandB)
alpha = 0.3  # nhỏ hơn thì mượt hơn
stamp_smooth = stamp_percent.ewm(alpha=alpha, adjust=False).mean()
fedavg_smooth = fedavg_percent.ewm(alpha=alpha, adjust=False).mean()

# Áp dụng rolling std cho fill_between
window_std = 5
stamp_std = stamp_percent.rolling(window=window_std, center=True, min_periods=1).std()
fedavg_std = fedavg_percent.rolling(window=window_std, center=True, min_periods=1).std()

# Vẽ biểu đồ
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(8, 6))

# FedAvg
plt.plot(step, fedavg_smooth, color='blue', label='FedAvg', linewidth=2.5)
plt.fill_between(step, fedavg_smooth - fedavg_std, fedavg_smooth + fedavg_std,
                 color='blue', alpha=0.3)

# STAMP
plt.plot(step, stamp_smooth, color='red', label='STAMP', linewidth=2.5)
plt.fill_between(step, stamp_smooth - stamp_std, stamp_smooth + stamp_std,
                 color='red', alpha=0.3)


# Cài đặt biểu đồ
plt.xlabel('Communication Rounds')
plt.ylabel('Percentage of Positive Cosine Similarities')
plt.grid(True)
plt.xlim(0, 250)
plt.ylim(0, 100)
plt.legend(loc= "lower right")
plt.tight_layout()
plt.show()
