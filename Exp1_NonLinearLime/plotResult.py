import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


def plot_metrics_comparison():
    results_dir = "results"
    file_names=["metrics_comparison_2025(10).csv", "metrics_comparison_2025(4).csv","metrics_comparison_2025(12).csv","metrics_comparison_2025(13).csv","metrics_comparison_2025(14).csv"]
    file_paths = [os.path.join(results_dir, name) for name in file_names]
    # 逐个加载文件
    dataframes = []
    for fp in file_paths:
        if os.path.exists(fp):
            print(f"[INFO] Found file: {fp}, loading data...")
            if fp.endswith(".csv"):
                df = pd.read_csv(fp)
            elif fp.endswith(".xlsx"):
                df = pd.read_excel(fp)
            else:
                raise ValueError("Unsupported file format. Use .csv or .xlsx.")
            dataframes.append(df)
        else:
            print(f"[ERROR] File {fp} not found. Please check the file path.")
            return

    if len(dataframes) < 5:
        print("[ERROR] Not all files were loaded. Please check the file paths.")
        return

    # 指标和解释方法定义
    metric_names = ["Fidelity", "Faithfulness", "Stability"]
    # 5个柱体对应的解释方法（英文标签）
    methods = [
        "Balanced Exhaustive Perturbation(num_token < 6)",
        "Binary Random Perturbation(num_token < 5)",
        "Balanced Exhaustive Perturbation(num_token < 5)",
        "Balanced Exhaustive Perturbation(num_token < 4)",
        "Balanced Exhaustive Perturbation(num_token < 3)"
    ]

    # 定义映射关系：(文件索引, 列后缀)
    # 说明：
    # 方法1：使用文件1中 LIME 结果
    # 方法2：使用文件1中 Nonlinear 结果
    # 方法3：使用文件2中 Nonlinear 结果
    # 方法4：使用文件3中 Nonlinear 结果
    # 方法5：使用文件4中 Nonlinear 结果
    mapping = [
        (0, "Nonlinear"),
        (1, "Nonlinear"),
        (2, "Nonlinear"),
        (3, "Nonlinear"),
        (4, "Nonlinear")
    ]

    # 计算每个指标在不同方法下的均值，结果存储在字典中
    # 格式：{ "Fidelity": { "LIME": value, "Model Output + FI": value, ... }, ... }
    metrics_means = {metric: {} for metric in metric_names}
    for method_idx, (file_idx, suffix) in enumerate(mapping):
        df = dataframes[file_idx]
        for metric in metric_names:
            col_name = f"{metric}_{suffix}"
            if col_name in df.columns:
                value = df[col_name].mean()
            else:
                print(f"[WARNING] Column '{col_name}' not found in file {file_paths[file_idx]}. Setting value to 0.")
                value = 0
            metrics_means[metric][methods[method_idx]] = value

    #colors = ['steelblue', 'coral', 'limegreen', 'orange', 'purple']
    colors = ['skyblue', 'cornflowerblue', 'dodgerblue', 'royalblue', 'navy']

    # 绘图：每个指标生成一个子图，每个子图中绘制 5 个柱体
    num_methods = len(methods)
    x = np.arange(num_methods)
    width = 0.6  # 柱体宽度

    fig, axes = plt.subplots(1, len(metric_names), figsize=(16, 5), dpi=100)
    plt.subplots_adjust(wspace=0.4)
    if len(metric_names) == 1:
        axes = [axes]

    for i, metric in enumerate(metric_names):
        ax = axes[i]
        values = [metrics_means[metric][m] for m in methods]
        # 为每个柱体使用不同颜色
        bars = ax.bar(x, values, width, color=colors, alpha=0.8)
        ax.set_title(metric, fontsize=13)
        ax.set_ylabel("Score")
        ax.set_xticks(x)
        ax.set_xticklabels([""] * num_methods)

        # 根据数据动态设置 y 轴范围
        min_val = min(values)
        max_val = max(values)
        if all(val >= 0 for val in values):
            ax.set_ylim(0, max_val + 0.2)
        else:
            y_margin = (max_val - min_val) * 0.2 if max_val != min_val else 0.1
            ax.set_ylim(min_val - y_margin, max_val + y_margin + 0.2)

        for j, bar in enumerate(bars):
            height = bar.get_height()
            # 如果是第四个方法（索引 3），则标注文字颜色为红色，否则为默认黑色
            color_txt = 'red' if j == 3 else 'black'
            fontweight = 'bold' if j == 1 else 'normal'
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.4f}",
                    ha='center', va='bottom', fontsize=10, color=color_txt, fontweight=fontweight)

    # 构建图例句柄，将每种颜色与对应方法关联
    handles = [Patch(facecolor=c, label=m) for c, m in zip(colors, methods)]

    leg = fig.legend(handles=handles, loc='upper right', ncol=1, fontsize=10)
    # 将图例中第四个方法的文字设为红色（索引 3）
    leg_texts = leg.get_texts()
    for idx, text in enumerate(leg_texts):
        if idx == 1:
            text.set_fontweight('bold')  # 第二个柱体加粗
        if idx == 3:
            text.set_color('red')  # 第四个柱体红色

    fig.suptitle("Metric Comparison Across Perturbation Strategies For NonLinearLime", fontsize=15)


    plt.tight_layout(rect=[0, 0, 1, 0.86])
    plt.show()


if __name__ == "__main__":
    plot_metrics_comparison()