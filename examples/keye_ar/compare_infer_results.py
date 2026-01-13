import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_geneval_from_csvs(csv_file_paths):
    """
    读取多个csv文件，提取Step和GenEval数据，绘制同一折线图
    :param csv_file_paths: 多个csv文件的路径列表
    """
    # 设置绘图样式，提升图表可读性
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 兼容英文显示
    plt.rcParams['figure.figsize'] = (12, 7)  # 图表尺寸
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', '*']

    # 遍历所有csv文件
    for idx, csv_path in enumerate(csv_file_paths):
        # 验证文件是否存在
        if not os.path.exists(csv_path):
            print(f"警告：文件 {csv_path} 不存在，已跳过")
            continue
        
        try:
            # 1. 读取csv文件
            df = pd.read_csv(csv_path)
            
            # 2. 验证必要列是否存在
            required_columns = ['Step', 'GenEval']
            if not all(col in df.columns for col in required_columns):
                print(f"警告：文件 {csv_path} 缺少必要列（Step/GenEval），已跳过")
                continue
            
            # 3. 数据预处理：去重（解决重复行问题）+ 按Step排序
            df_clean = df[required_columns].drop_duplicates(subset='Step').sort_values(by='Step')
            df_clean = df_clean[df_clean['Step'] >= 6000]

            print(df_clean)

            # 4. 提取绘图数据
            steps = df_clean['Step'].values
            geneval_scores = df_clean['GenEval'].values
            
            # 5. 绘制折线图（每个文件对应不同样式）
            file_name = os.path.basename(csv_path)  # 获取文件名作为图例
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            plt.plot(steps, geneval_scores, label=file_name, color=color, marker=marker, 
                     markersize=4, linewidth=2, alpha=0.8)
        
        except Exception as e:
            print(f"警告：处理文件 {csv_path} 时出错，已跳过，错误信息：{str(e)}")
            continue
    
    # 6. 图表美化和标注
    plt.title('Step vs GenEval Scores Comparison (Multiple CSV Files)', fontsize=16, pad=20)
    plt.xlabel('Step', fontsize=14, labelpad=10)
    plt.ylabel('GenEval Score', fontsize=14, labelpad=10)
    plt.grid(True, alpha=0.3, linestyle='--')  # 添加网格线
    plt.legend(fontsize=12, loc='best')  # 显示图例
    plt.tick_params(axis='both', which='major', labelsize=12)  # 调整刻度字体大小
    
    # 7. 保存图表（高分辨率）+ 显示图表
    output_png = 'geneval_step_comparison.png'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"图表已保存为：{output_png}")
    plt.show()


# python3 examples/keye_ar/compare_infer_results.py /mmu_mllm_hdd_2/lingzhixin/output/MuseV2/sana/ar_dit_23p/exp22_ar_dit_324tokens_1e-4_reproduce/GenEval_scores.csv /mmu_mllm_hdd_2/lingzhixin/output/MuseV2/sana/ar_dit_23p/exp26_ar_dit_324tokens_1e-4_reproduce_with_pos/GenEval_scores.csv 
# 
if __name__ == "__main__":
    # 解析命令行参数，支持传入多个csv文件
    parser = argparse.ArgumentParser(description='绘制多个csv文件的Step vs GenEval折线图')
    parser.add_argument('csv_files', nargs='+', help='一个或多个包含Step和GenEval列的csv文件路径')
    args = parser.parse_args()
    
    # 调用绘图函数
    plot_geneval_from_csvs(args.csv_files)