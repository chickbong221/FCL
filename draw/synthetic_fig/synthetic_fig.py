import matplotlib.pyplot as plt
import numpy as np

# Trục x: các mức độ dị biệt dữ liệu
x = ['α=100', 'α=10', 'α=1.0', 'α=0.1']
x_ticks = np.arange(len(x))

# Dữ liệu xấp xỉ từ hình ảnh (4 datasets × 10 phương pháp)
data = {
    'FedAvg': [[52, 50, 45, 38], [40, 35, 28, 18], [80, 78, 76, 70], [65, 60, 55, 45]],
    'FedProx': [[54, 52, 47, 39], [42, 38, 30, 21], [79, 77, 75, 69], [66, 61, 56, 46]],
    'FL+EWC': [[56, 54, 50, 42], [43, 40, 33, 26], [82, 80, 78, 72], [67, 62, 58, 48]],
    'FL+SI': [[55, 53, 48, 41], [44, 41, 35, 28], [83, 81, 79, 73], [68, 63, 59, 49]],
    'Re-Fed': [[53, 51, 47, 40], [41, 37, 32, 25], [81, 79, 77, 71], [66, 61, 57, 47]],
    'FedCIL': [[58, 55, 50, 44], [46, 42, 36, 30], [84, 82, 80, 74], [69, 64, 60, 50]],
    'FOT': [[60, 58, 53, 47], [47, 44, 39, 33], [85, 84, 82, 76], [70, 65, 61, 51]],
    'FedWeIT': [[62, 60, 55, 48], [48, 45, 40, 35], [86, 85, 83, 77], [71, 66, 62, 52]],
    'FedSSI': [[65, 63, 58, 52], [49, 47, 43, 38], [88, 87, 85, 80], [72, 67, 63, 53]],
}

colors = {
    'FedAvg': 'blue',
    'FedProx': 'orange',
    'FL+EWC': 'brown',
    'FL+SI': 'yellowgreen',
    'Re-Fed': 'pink',
    'FedCIL': 'green',
    'FOT': 'red',
    'FedWeIT': 'purple',
    'FedSSI': 'black'
}
linestyles = {
    'FedSSI': 'dashdot'
}
markers = {
    'FedSSI': 'o'
}

titles = ['CIFAR10 Dataset', 'CIFAR100 Dataset', 'Digit10 Dataset', 'Office31 Dataset']
y_lims = [(35, 67), (16, 50), (68, 90), (44, 75)]
x_lims = (0,3)

fig, axs = plt.subplots(1, 4, figsize=(20, 4), sharex=False)

for i, ax in enumerate(axs):
    for name, values in data.items():
        line, = ax.plot(x_ticks, values[i], label=name, color=colors.get(name, None),
                        linestyle=linestyles.get(name, '-'), marker=markers.get(name, ''))
    ax.set_title(titles[i])
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x)
    ax.set_ylim(y_lims[i])
    ax.set_xlim(x_lims)
    ax.set_xlabel("Data Heterogeneity")
    if i == 0:
        ax.set_ylabel("Test Accuracy")
    ax.grid(True)

# Chỉnh legend ở trên giữa
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.0),
    ncol=9,
    fontsize=10,
    frameon=True,              # Hiển thị khung
    fancybox=True,             # Làm tròn góc khung
    shadow=True,              # Không đổ bóng
    framealpha=1.0,            # Độ trong suốt của khung (1.0 là đậm)
    edgecolor='gray',          # Màu viền khung
    facecolor='white'          # Màu nền khung
)
# Tiêu đề chính (suptitle) đặt ở dưới
fig.text(0.5, -0.12, "Figure 3. Performance w.r.t data heterogeneity $\\alpha$ for four datasets.", 
         ha='center', fontsize=12, style='italic')

fig.tight_layout(rect=[0, 0, 1, 0.93])  # Chừa không gian phía trên cho legend
plt.show()
