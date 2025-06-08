import logging
import os
import pandas as pd

def init_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    return logger 

def load_evaluation_results(model_id, quant_method, dataset, full_c4, seqlen, nsamples, results_folder=None):
    """
    加载指定参数下的评测结果CSV文件，返回DataFrame或None。
    """
    if results_folder is None:
        # 默认结果文件夹与evaluate.py一致
        seqlen_str = str(seqlen)
        result_suffix = f"eval_results_quant-{quant_method}_data-{dataset}_fullc4-{full_c4}_seqlen-{seqlen_str}_nsamples-{nsamples}"
        results_folder = os.path.join(model_id, result_suffix)
    # 构造CSV文件名
    result_suffix = f"quant-{quant_method}_data-{dataset}_fullc4-{full_c4}_seqlen-{seqlen}_nsamples-{nsamples}"
    csv_path = os.path.join(results_folder, f'evaluation_results_{result_suffix}.csv')
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path) 