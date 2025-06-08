import matplotlib.pyplot as plt
import os
import pandas as pd

def plot_memory_and_time(auto_results, seqlen_result_suffix, results_folder):
    plot_path = os.path.join(results_folder, f'performance_metrics_{seqlen_result_suffix}.png')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.plot(auto_results['Memory Usage'], label='Memory Usage')
    ax1.set_title(f'GPU Memory Usage Over Time')
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Memory Usage (MB)')
    ax1.grid(True)
    ax1.legend()
    ax2.plot(auto_results['Times'], label='Processing Time')
    ax2.set_title('Token Processing Time')
    ax2.set_xlabel('Token Position')
    ax2.set_ylabel('Time (seconds)')
    ax2.grid(True)
    ax2.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_path

def plot_ppl_comparison(summary_df, results_folder):
    ppl_plot_path = os.path.join(results_folder, 'seqlen_ppl_comparison.png')
    plt.figure(figsize=(10, 6))
    plt.plot(summary_df['Sequence Length'], summary_df['Auto PPL'], marker='o', label='Auto PPL')
    plt.plot(summary_df['Sequence Length'], summary_df['Strict PPL'], marker='o', label='Strict PPL')
    plt.xlabel('Sequence Length')
    plt.ylabel('PPL')
    plt.title('PPL vs Sequence Length')
    plt.legend()
    plt.grid(True)
    plt.savefig(ppl_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return ppl_plot_path

def plot_acc_comparison(summary_df, results_folder):
    acc_plot_path = os.path.join(results_folder, 'seqlen_acc_comparison.png')
    plt.figure(figsize=(10, 6))
    plt.plot(summary_df['Sequence Length'], summary_df['Auto Accuracy'], marker='o', label='Auto Accuracy (%)')
    plt.plot(summary_df['Sequence Length'], summary_df['Strict Accuracy'], marker='o', label='Strict Accuracy (%)')
    plt.xlabel('Sequence Length')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Sequence Length')
    plt.legend()
    plt.grid(True)
    plt.savefig(acc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return acc_plot_path 