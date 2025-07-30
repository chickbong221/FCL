import pandas as pd
import matplotlib.pyplot as plt

# Đọc file CSV
df = pd.read_csv('client_9.csv')

# Step từ cột đầu tiên
step = df.iloc[:, 0]

# Lấy dữ liệu từ cột thứ 4 (index 3) và cột thứ 7 (index 6)
data_col4 = df.iloc[:, 3]
data_col7 = df.iloc[:, 6]

# Làm mượt EMA trực tiếp trên dữ liệu gốc
alpha = 0.05
data_col4_smooth = data_col4.ewm(alpha=alpha, adjust=False).mean()
data_col7_smooth = data_col7.ewm(alpha=alpha, adjust=False).mean()

# Rolling std cho fill_between
window_std = 5
std_col4 = data_col4.rolling(window=window_std, center=True, min_periods=1).std()
std_col7 = data_col7.rolling(window=window_std, center=True, min_periods=1).std()

# Vẽ biểu đồ
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(8, 6))

# Đường từ cột 7 (FedAvg)
plt.plot(step, data_col7_smooth, color='blue', label='FedAvg', linewidth=2.5)
plt.fill_between(step,
                 data_col7_smooth - std_col7,
                 data_col7_smooth + std_col7,
                 color='blue', alpha=0.3)

# Đường từ cột 4 (STAMP)
plt.plot(step, data_col4_smooth, color='red', label='STAMP', linewidth=2.5)
plt.fill_between(step,
                 data_col4_smooth - std_col4,
                 data_col4_smooth + std_col4,
                 color='red', alpha=0.3)

# Cài đặt biểu đồ
plt.xlabel('Communication Rounds')
plt.ylabel('Cosine Similarity Value')
plt.grid(True)
plt.xlim(0, 250)
plt.ylim(-0.3, 0.3)
plt.legend()
plt.tight_layout()
plt.show()
