import os
import logging
import torch
import gc
import json
from transformers import AutoModelForCausalLM, AutoConfig, LlamaForCausalLM
from awq import AutoAWQForCausalLM
import sys
# from squeezellm.modelutils import *
# from squeezellm.quant import *

def create_results_folder(model_id, result_suffix):
    results_dir = os.path.join(model_id, result_suffix)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def get_optimal_device():
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name()
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logging.info(f"Using GPU: {gpu_name} with {total_memory:.2f}GB total memory")
        return device
    elif torch.backends.mps.is_available():
        device = "mps"
        logging.info("Using Apple Silicon (MPS)")
        return device
    else:
        device = "cpu"
        logging.info("No GPU found, using CPU")
        return device

def load_model(model_path, device, quant_method="base", base_model_path=None, weight_path=None, quant_config_path=None, wbits=4, include_sparse=False, topX=10):
    try:
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        logging.info(f"Loading model from {model_path} with quantization method: {quant_method}")
        if quant_method == "base":
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                use_cache=True,
                low_cpu_mem_usage=True
            )
            model = model.eval()
        elif quant_method == "awq":
            model = AutoAWQForCausalLM.from_quantized(
                model_path,
                device=device,
                use_cache=True,
                low_cpu_mem_usage=True
            )
            model = model.eval()
        elif quant_method == "gptq":
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                trust_remote_code=True
            )
            model = model.eval()
        elif quant_method == "qllm":
            from qllm.modeling.base import AutoQuantizedModelForCausalLM
            if os.path.exists(model_path) or '/' in model_path:
                logging.info(f"Loading QLLM quantized model from {model_path}")
                model = AutoQuantizedModelForCausalLM.from_quantized(
                    model_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    device_map="auto"
                )
                model = model.to(device)
                model = model.eval()
            elif base_model_path is not None:
                logging.info(f"Quantizing model {base_model_path} with QLLM using {wbits} bits")
                from qllm.auto_model_quantization import AutoModelQuantization
                quantizer = AutoModelQuantization()
                model = quantizer.api_quantize(
                    model_or_model_path=base_model_path,
                    method='gptq',
                    wbits=wbits,
                    groupsize=128
                )
                model = model.to(device)
                model = model.eval()
            else:
                raise ValueError("For QLLM quantization, either model_path or base_model_path must be specified")
        elif quant_method == "vptq":
            import vptq
            model = vptq.AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
            )
            model = model.eval()
            logging.info("VPTQ model loaded successfully")
        elif quant_method == "sqllm":
            if base_model_path is None or weight_path is None:
                raise ValueError("For sqllm quant_method, --sqllm_base_model and --sqllm_weight must be specified.")
            
            # Use the load_quant function to properly load SqueezeLLM models
            logging.info(f"Loading SqueezeLLM model: base={base_model_path}, weights={weight_path}, wbits={wbits}")
            
            # Load the model using proper SqueezeLLM loading method
            if "xgen" in base_model_path or "opt" in base_model_path or \
               ("vicuna" in base_model_path and "v1.3" in base_model_path) or \
               "llama-2" in base_model_path or "mistral" in base_model_path:
                config = AutoConfig.from_pretrained(base_model_path)
                model = AutoModelForCausalLM.from_config(config)
            else:
                model = LlamaForCausalLM.from_pretrained(base_model_path, torch_dtype="auto")
            
            model = model.eval()
            layers = find_layers(model)
            
            # Exclude lm_head from quantization
            if "lm_head" in layers:
                del layers["lm_head"]
            
            # Set up quantization with lookup tables
            make_quant_lut(
                model, layers, wbits, include_sparse=include_sparse, numvals=None, topX=topX
            )
            
            # Load the quantized weights
            logging.info(f"Loading quantized weights from {weight_path}")
            state_dict = torch.load(weight_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            
            # Load configuration if provided
            if quant_config_path is not None:
                if os.path.exists(quant_config_path):
                    with open(quant_config_path, "r") as f:
                        quant_config = json.load(f)
                    logging.info(f"Loaded quant_config: {quant_config}")
                    # Use configuration values if available
                    if "wbits" in quant_config:
                        logging.info(f"Using wbits from config: {quant_config['wbits']}")
                else:
                    logging.warning(f"quant_config.json not found at {quant_config_path}")
            
            model.seqlen = 2048
            model = model.to(device)
            logging.info("SqueezeLLM model loaded successfully")

            
        elif quant_method == "duquant":
            from transformers import PreTrainedTokenizerFast
            model = LlamaForCausalLM.from_pretrained(model_path)
            tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
            model = model.to(device)
            model = model.eval()
            logging.info("DuQuant model loaded successfully")

        else:
            raise ValueError(f"Unsupported quantization method: {quant_method}")
        

        logging.info("Model loaded successfully")
        return model
    
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise




# squeezellm:
import torch
import torch.nn as nn


# function to find layers in the network (either for packing or for replacement)
def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res

import numpy as np
import torch
import torch.nn as nn
import math
import quant_cuda


def round_to_nearest_pole_sim(w, poles):
    """
    w: weight values (1d vector)
    poles: tuple of values

    Round the numbers in w to the nearest value in poles.
    """
    stack = []
    for c in poles:
        diff = (w - c).abs()
        stack.append(diff)
    diff = torch.stack(stack)
    idx = diff.argmin(axis=0)
    aug = 0
    for i, c in enumerate(poles):
        aug += (idx == i) * c
    return aug


# drop-in layer replacement class
class QuantLinearLUT(nn.Module):
    def __init__(
        self,
        bits,
        infeatures,
        outfeatures,
        bias,
        include_sparse=False,
        numvals=0,
        topX=0,
        balanced=False,
        num_nonzero_per_thread=10,
    ):
        super().__init__()
        if bits not in [3, 4]:
            raise NotImplementedError("Only 3 and 4 bits is supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits

        self.register_buffer(
            "qweight",
            torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32),
        )
        if bias:
            self.include_bias = True
            self.register_buffer("bias", torch.zeros((outfeatures)))
        else:
            self.include_bias = False
            self.bias = None
        self.register_buffer(
            "lookup_table",
            torch.zeros((outfeatures, 2**self.bits), dtype=torch.float32),
        )

        self.include_sparse = include_sparse
        self.numvals = numvals
        self.topX = topX
        if numvals > 0:
            self.register_buffer(
                "rows", torch.zeros(outfeatures + 1, dtype=torch.int32)
            )
            self.register_buffer("cols", torch.zeros(numvals, dtype=torch.int32))
            self.register_buffer("vals", torch.zeros(numvals, dtype=torch.float32))

            print("self.rows: ", self.rows)
        if topX > 0:
            self.register_buffer(
                "full_rows", torch.zeros((infeatures, topX), dtype=torch.float32)
            )
            self.register_buffer(
                "full_row_indices", torch.zeros(topX, dtype=torch.int32)
            )

        self.balanced = balanced

        if include_sparse and balanced and numvals > 0:
            print("use num_nonzero_per_thread")
            self.num_threads = int(
                (numvals + num_nonzero_per_thread - 1) / num_nonzero_per_thread
            )
            self.num_threads = 128 * math.ceil(
                self.num_threads / 128
            )  # round up to nearest factor of blocksize = 128
            self.register_buffer(
                "startrows", torch.zeros(self.num_threads, dtype=torch.int32)
            )
            print("self.num_threads : ", self.num_threads)

    def pack2(self, linear, lookup_table, include_sparse, num_nonzero_per_thread=-1):
        if self.include_bias:  # linear.bias is not None:
            self.bias = linear.bias.clone()  # todo: check this condition

        # self.lookup_table = lookup_table.float()
        lut, outliers = lookup_table

        # handle dense matrix
        intweight = linear.weight.data.clone()

        if include_sparse:
            outliers = outliers.to_dense()

        # get zero mapping
        num_channels = len(lut)
        for channel in range(num_channels):
            centroid, indices = lut[channel][0]  # last 0 is for group 0
            intweight[channel] = torch.from_numpy(indices)
            self.lookup_table[channel] = torch.from_numpy(centroid)

            if include_sparse:
                zero_mapping = round_to_nearest_pole_sim(torch.zeros(1), centroid)
                nonzero_vals = torch.nonzero(outliers[channel])

                outliers_channel = outliers[channel]
                outliers_channel[nonzero_vals] -= zero_mapping
                outliers[channel] = outliers_channel

        if include_sparse:
            outliers = outliers.to_sparse(layout=torch.sparse_csr)

            # save sparse matrix (already in CSR)
            self.register_buffer("rows", outliers.crow_indices().to(torch.int32))
            self.register_buffer("cols", outliers.col_indices().to(torch.int32))
            self.register_buffer("vals", outliers.values().to(torch.float32))

            # self.balanced
            if self.balanced:
                self.numvals = self.vals.shape[0]
                print("self.numvals: ", self.numvals)
                print("self.rows: ", self.rows.shape[0])

                self.num_threads = int(
                    (self.numvals + num_nonzero_per_thread - 1)
                    / num_nonzero_per_thread
                )
                self.num_threads = 128 * math.ceil(
                    self.num_threads / 128
                )  # round up to nearest factor of blocksize = 128

                nnz_per_thread = int(
                    (self.numvals + self.num_threads - 1) / self.num_threads
                )
                start_rows = torch.zeros(self.num_threads, dtype=torch.int32)

                print("self.num_threads: ", self.num_threads)
                print("nnz_per_thread: ", nnz_per_thread)

                minidx = 0
                for i in range(0, self.num_threads):
                    tmpmin = minidx
                    for j in range(minidx, self.outfeatures):
                        if nnz_per_thread * i > self.numvals:
                            start_rows[i] = -1
                            break
                        elif self.rows[j] < nnz_per_thread * i:
                            start_rows[i] = j
                            tmpmin = j
                        else:
                            break
                    minidx = tmpmin

                self.register_buffer("startrows", start_rows)

        intweight = intweight.to(torch.int)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        qweight = np.zeros(
            (intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32
        )
        i = 0
        row = 0
        while row < qweight.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i))
                i += 10
                qweight[row] |= intweight[i] << 30
                row += 1
                qweight[row] |= (intweight[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row] |= intweight[i] << 31
                row += 1
                qweight[row] |= (intweight[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 2)
                i += 10
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

    # replacement forward pass
    def forward(self, x):
        if x.shape[-1] == x.numel():
            outshape = list(x.shape)
            if self.bias is not None:
                y = self.bias.clone()
                outshape[-1] = self.bias.numel()
            else:
                y = torch.zeros((self.outfeatures), device="cuda", dtype=torch.float32)
                outshape[-1] = self.outfeatures
            dtype = x.dtype

            if self.bits == 3:
                x = x.float()
                if self.include_sparse and self.topX > 0:
                    quant_cuda.vecquant3matmul_spmv_hybrid_nuq_perchannel(
                        self.rows,
                        self.cols,
                        self.vals,
                        x,
                        self.full_rows,
                        self.full_row_indices,
                        y,
                        self.outfeatures,
                        self.qweight,
                        self.lookup_table,
                    )
                elif self.include_sparse and self.balanced:
                    quant_cuda.vecquant3matmul_spmv_balanced_nuq_perchannel(
                        self.rows,
                        self.cols,
                        self.startrows,
                        self.vals,
                        x,
                        y,
                        self.qweight,
                        self.lookup_table,
                        self.outfeatures,
                        self.num_threads,
                        self.numvals,
                    )
                elif self.include_sparse:
                    quant_cuda.vecquant3matmul_spmv_nuq_perchannel(
                        self.rows,
                        self.cols,
                        self.vals,
                        x,
                        y,
                        self.outfeatures,
                        self.qweight,
                        self.lookup_table,
                    )
                else:
                    quant_cuda.vecquant3matmul_nuq_perchannel(
                        x, self.qweight, y, self.lookup_table
                    )
            elif self.bits == 4:
                x = x.float()
                if self.include_sparse and self.topX > 0:
                    quant_cuda.vecquant4matmul_spmv_hybrid_nuq_perchannel(
                        self.rows,
                        self.cols,
                        self.vals,
                        x,
                        self.full_rows,
                        self.full_row_indices,
                        y,
                        self.outfeatures,
                        self.qweight,
                        self.lookup_table,
                    )
                elif self.include_sparse and self.balanced:
                    quant_cuda.vecquant4matmul_spmv_balanced_nuq_perchannel(
                        self.rows,
                        self.cols,
                        self.startrows,
                        self.vals,
                        x,
                        y,
                        self.qweight,
                        self.lookup_table,
                        self.outfeatures,
                        self.num_threads,
                        self.numvals,
                    )
                elif self.include_sparse:
                    quant_cuda.vecquant4matmul_spmv_nuq_perchannel(
                        self.rows,
                        self.cols,
                        self.vals,
                        x,
                        y,
                        self.outfeatures,
                        self.qweight,
                        self.lookup_table,
                    )
                else:
                    quant_cuda.vecquant4matmul_nuq_perchannel(
                        x, self.qweight, y, self.lookup_table
                    )

            y = y.to(dtype)
            return y.reshape(outshape)
        else:
            out_shape = x.shape[:-1] + (self.outfeatures,)
            x = x.reshape(-1, x.shape[-1])
            out = torch.zeros(
                (x.shape[0], self.outfeatures), device="cuda", dtype=torch.float32
            )
            dtype = x.dtype
            if self.bits == 3:
                x = x.float()
                if self.include_sparse and self.topX > 0:
                    quant_cuda.vecquant3matmul_spmv_hybrid_nuq_perchannel_batched(
                        self.rows,
                        self.cols,
                        self.vals,
                        x,
                        self.full_rows,
                        self.full_row_indices,
                        out,
                        self.outfeatures,
                        self.qweight,
                        self.lookup_table,
                    )
                elif self.include_sparse:
                    quant_cuda.vecquant3matmul_spmv_nuq_perchannel_batched(
                        self.rows,
                        self.cols,
                        self.vals,
                        x,
                        out,
                        self.outfeatures,
                        self.qweight,
                        self.lookup_table,
                    )
                else:
                    quant_cuda.vecquant3matmul_nuq_perchannel_batched(
                        x, self.qweight, out, self.lookup_table
                    )
            elif self.bits == 4:
                x = x.float()
                if self.include_sparse and self.topX > 0:
                    quant_cuda.vecquant4matmul_spmv_hybrid_nuq_perchannel_batched(
                        self.rows,
                        self.cols,
                        self.vals,
                        x,
                        self.full_rows,
                        self.full_row_indices,
                        out,
                        self.outfeatures,
                        self.qweight,
                        self.lookup_table,
                    )
                elif self.include_sparse:
                    quant_cuda.vecquant4matmul_spmv_nuq_perchannel_batched(
                        self.rows,
                        self.cols,
                        self.vals,
                        x,
                        out,
                        self.outfeatures,
                        self.qweight,
                        self.lookup_table,
                    )
                else:
                    quant_cuda.vecquant4matmul_nuq_perchannel_batched(
                        x, self.qweight, out, self.lookup_table
                    )
            out = out.to(dtype)
            out = out.reshape(out_shape)
            out = out + self.bias if self.bias is not None else out
            return out


def make_quant_lut(
    module,
    names,
    bits,
    name="",
    include_sparse=False,
    numvals=None,
    topX=0,
    balanced=False,
    num_nonzero_per_thread=10,
):
    if isinstance(module, QuantLinearLUT):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + "." + attr if name != "" else attr
        if name1 in names:
            if numvals is not None:
                print("name1 ", name1)
                num = numvals[name1]
            else:
                num = 0
            delattr(module, attr)
            setattr(
                module,
                attr,
                QuantLinearLUT(
                    bits,
                    tmp.in_features,
                    tmp.out_features,
                    tmp.bias is not None,
                    include_sparse=include_sparse,
                    numvals=num,
                    topX=topX,
                    balanced=balanced,
                    num_nonzero_per_thread=num_nonzero_per_thread,
                ),
            )
    for name1, child in module.named_children():
        make_quant_lut(
            child,
            names,
            bits,
            name + "." + name1 if name != "" else name1,
            include_sparse=include_sparse,
            numvals=numvals,
            topX=topX,
            balanced=balanced,
            num_nonzero_per_thread=num_nonzero_per_thread,
        )
