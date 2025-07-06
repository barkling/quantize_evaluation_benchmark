import argparse
import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from model_loader import create_results_folder, get_optimal_device, load_model
from data_utils import get_wikitext2, get_c4
from evaluation import evaluate_autoregressive, evaluate_strict, evaluate_sliding_window
from plot_utils import plot_memory_and_time, plot_ppl_comparison, plot_acc_comparison
from utils import init_logging, load_evaluation_results
import os

def main():
    logger = init_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="Model path or name")
    parser.add_argument("--quant_method", type=str, default="base", 
                        choices=["base", "awq", "gptq", "sqllm", "qllm", "vptq", "duquant"], 
                        help="Quantization method")
    parser.add_argument("--sqllm_base_model", type=str, default=None, 
                        help="Base model path for SqueezeLLM or QLLM quantization")
    parser.add_argument("--sqllm_weight", type=str, default=None, help="Quantized weight path for SqueezeLLM (only for sqllm)")
    parser.add_argument("--sqllm_quant_config", type=str, default=None, help="Quant config json path for SqueezeLLM (only for sqllm)")
    parser.add_argument("--wbits", type=int, default=4, help="Bit-width for SqueezeLLM quantization")
    parser.add_argument("--include_sparse", action="store_true", help="Whether to include sparse quantization for SqueezeLLM")
    parser.add_argument("--topX", type=int, default=10, help="TopX value for SqueezeLLM quantization")
    parser.add_argument("--dataset", type=str, default="wikitext2", choices=["wikitext2", "c4"], help="Dataset to evaluate on")
    parser.add_argument("--seqlen", type=int, default=2048, help="Sequence length (used if --seqlen_list not set)")
    parser.add_argument("--seqlen_list", type=int, nargs='+', default=None, help="List of sequence lengths to evaluate, e.g. --seqlen_list 512 1024 2048")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of samples")
    parser.add_argument("--full_c4", action="store_true", help="Use full C4 dataset instead of fragments")
    parser.add_argument("--use_cached_results", action="store_true", help="If set, load results from cache if available.")
    args = parser.parse_args()

    device = get_optimal_device()
    result_suffix = f"eval_results_quant-{args.quant_method}_data-{args.dataset}_fullc4-{args.full_c4}_nsamples-{args.nsamples}"
    results_folder = create_results_folder(args.model_id, result_suffix)

    try:
        model = None
        tokenizer = None
        seqlen_list = args.seqlen_list if args.seqlen_list is not None else [args.seqlen]
        all_results = []
        for seqlen in seqlen_list:
            logger.info(f"\n==== Evaluating for seqlen={seqlen} ====")
            seqlen_result_suffix = f"quant-{args.quant_method}_data-{args.dataset}_fullc4-{args.full_c4}_seqlen-{seqlen}_nsamples-{args.nsamples}"
            cached_df = None
            if args.use_cached_results:
                cached_df = load_evaluation_results(
                    args.model_id, args.quant_method, args.dataset, args.full_c4, seqlen, args.nsamples, results_folder=results_folder
                )
            if cached_df is not None:
                logger.info(f"Loaded cached results for seqlen={seqlen}")
                results = cached_df.iloc[0].to_dict()
                all_results.append(results)
                print(f"\nEvaluation Results for seqlen={seqlen} (CACHED):")
                for key, value in results.items():
                    print(f"{key}: {value}")
                continue
            if model is None:
                if args.quant_method == "sqllm":
                    model = load_model(
                        args.model_id, 
                        device, 
                        args.quant_method, 
                        base_model_path=args.sqllm_base_model, 
                        weight_path=args.sqllm_weight, 
                        quant_config_path=args.sqllm_quant_config,
                        wbits=args.wbits,
                        include_sparse=args.include_sparse,
                        topX=args.topX
                    )
                elif args.quant_method == "qllm":
                    model = load_model(
                        args.model_id, 
                        device, 
                        args.quant_method, 
                        base_model_path=args.sqllm_base_model,
                        wbits=args.wbits
                    )
                else:
                    model = load_model(args.model_id, device, args.quant_method)
            if tokenizer is None:
                tokenizer_path = args.sqllm_base_model if args.quant_method == "sqllm" else args.model_id
                if args.quant_method == "duquant":
                    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            if args.dataset == "wikitext2":
                testdata = get_wikitext2(args.nsamples, 0, seqlen, args.model_id)
            else:
                testdata = get_c4(args.nsamples, 0, seqlen, args.model_id, args.full_c4)
            auto_results = evaluate_autoregressive(model, testdata, tokenizer, device, seqlen)
            strict_ppl, strict_acc = evaluate_strict(model, testdata, tokenizer, device, max_samples=args.nsamples, seqlen=seqlen)
            sliding_ppl, sliding_acc = evaluate_sliding_window(model, testdata, tokenizer, device, window_size=seqlen, stride=1, max_samples=32)
            results = {
                'Model': args.model_id,
                'Quantization': args.quant_method,
                'Dataset': args.dataset,
                'Full C4': args.full_c4,
                'Sequence Length': seqlen,
                'Auto PPL': auto_results['PPL'],
                'Strict PPL': strict_ppl,
                'Sliding PPL': sliding_ppl,
                'Auto Accuracy': auto_results['Accuracy'],
                'Strict Accuracy': strict_acc * 100,
                'Sliding Accuracy': sliding_acc * 100,
                'Max Memory (MB)': auto_results['Max Memory (MB)'],
                'Median Time (s)': auto_results['Median Time (s)']
            }
            all_results.append(results)
            csv_path = os.path.join(results_folder, f'evaluation_results_{seqlen_result_suffix}.csv')
            results_df = pd.DataFrame([results])
            results_df.to_csv(csv_path, index=False)
            plot_memory_and_time(auto_results, seqlen_result_suffix, results_folder)
            print(f"\nEvaluation Results for seqlen={seqlen}:")
            for key, value in results.items():
                print(f"{key}: {value}")
        summary_df = pd.DataFrame(all_results)
        summary_csv = os.path.join(results_folder, f'summary_seqlen_comparison.csv')
        summary_df.to_csv(summary_csv, index=False)
        ppl_plot_path = plot_ppl_comparison(summary_df, results_folder)
        acc_plot_path = plot_acc_comparison(summary_df, results_folder)
        print(f"\nAll sequence length results saved to {summary_csv}")
        print(f"PPL comparison plot saved to {ppl_plot_path}")
        print(f"Accuracy comparison plot saved to {acc_plot_path}")
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 