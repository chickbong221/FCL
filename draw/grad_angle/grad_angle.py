import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ CSV
df = pd.read_csv('grad_angle.csv')

# Lấy các cột MAX (bỏ cột đầu tiên là Step)
max_columns = [col for col in df.columns if col.endswith('__MAX')]

# Lấy cột Step (giả sử là cột đầu tiên)
x = df.iloc[:, 0]

# Cấu hình cho EMA và std
ema_span = 80  # độ mượt
std_window = 5  # cửa sổ để tính std

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
for col in max_columns:
    ema = df[col].ewm(span=ema_span, adjust=False).mean()
    std = df[col].rolling(window=std_window, min_periods=1).std()

    plt.plot(x, ema, label=col.replace('__MAX', ''), linewidth=2.5)
    plt.fill_between(x, ema - std, ema + std, alpha=0.2)

plt.xlabel('Step')
plt.ylabel('Gradient Angle')
plt.title('EMA of Gradient Angles with Std Region')
plt.legend()
plt.grid(True)
plt.xlim(left=0, right=400)
plt.ylim(bottom=0, top=0.06)
plt.tight_layout()
plt.show()
