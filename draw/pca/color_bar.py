import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

def draw_color_column_svg(output_path="color_column.svg"):
    cmap = mpl.cm.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(0.5, 9))  # Kích thước hợp lý để xuất SVG

    for i in range(10):
        color = cmap(i)
        rect = patches.Rectangle((0, i), 1, 1, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(1.1, i + 0.5, f"{i+1}", va='center', ha='left', fontsize=10)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 10)
    ax.axis('off')  # Tắt trục

    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()
    print(f"Đã lưu cột màu SVG tại: {output_path}")

draw_color_column_svg("color_column.svg")
