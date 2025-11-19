"""Quantize a model on WikiText-2, compare PPL with original, and save to CSV."""

import torch
import csv
import os
import gc
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from nanomodel import AutoNanoModel, QuantizeConfig

pretrained_model_id = "Qwen/Qwen3-4B-Instruct-2507"
quantized_model_id = "/home/sd24191/git_project/QModel-Demo/quantized_models/Qwen3-4B-Instruct-2507-GPTQ-4bit"
csv_output_file = "ppl_comparison.csv"


def get_wikitext2(tokenizer, nsamples, seqlen):
    """Prepare a tokenized subset of WikiText-2 with minimum length filtering."""
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(
        lambda x: len(x["text"]) >= seqlen
    )

    return [tokenizer(example["text"]) for example in traindata.select(range(nsamples))]


def save_ppl_to_csv(model_name, ppl, filename):
    """Append PPL result to a CSV file."""
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        # Write header if file is new
        if not file_exists:
            writer.writerow(['Model_ID', 'Perplexity'])
        
        writer.writerow([model_name, f"{ppl:.4f}"])
    print(f"Saved PPL for {model_name} to {filename}")


@torch.inference_mode()
def calculate_avg_ppl(model, tokenizer):
    """Compute the average perplexity of the model on WikiText-2."""
    from nanomodel.utils.perplexity import Perplexity

    ppl = Perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_path="wikitext",
        dataset_name="wikitext-2-raw-v1",
        split="train",
        text_column="text",
    )

    all_ppl = ppl.calculate(n_ctx=512, n_batch=512)

    # average ppl
    avg = sum(all_ppl) / len(all_ppl)

    return avg


def main():
    """Execute comparison pipeline: Baseline PPL -> Quantize -> Quantized PPL -> CSV Report."""
    # Setup basic logging
    import logging
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id, use_fast=True)

    # ---------------------------------------------------------
    # 1. Calculate Baseline PPL (Original Model)
    # ---------------------------------------------------------
    print(f"--- Loading Original Model: {pretrained_model_id} ---")
    # Load using standard transformers to ensure correct baseline performance on GPU
    original_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_id, 
        torch_dtype=torch.float16, 
        device_map=device,
        trust_remote_code=True
    )
    
    original_ppl = calculate_avg_ppl(original_model, tokenizer)
    print(f"Original Model PPL: {original_ppl}")
    save_ppl_to_csv(pretrained_model_id, original_ppl, csv_output_file)

    # Cleanup to free VRAM for quantization
    del original_model
    torch.cuda.empty_cache()
    gc.collect()

    # ---------------------------------------------------------
    # 2. Quantization Process
    # ---------------------------------------------------------
    print(f"--- Starting Quantization for: {pretrained_model_id} ---")
    traindataset = get_wikitext2(tokenizer, nsamples=256, seqlen=1024)

    quantize_config = QuantizeConfig(
        bits=4,  # quantize model to 4-bit
        group_size=128,  # it is recommended to set the value to 128
    )

    # Load un-quantized model (NanoModel forces CPU load for quant preparation)
    model = AutoNanoModel.load(pretrained_model_id, quantize_config)

    # Quantize
    model.quantize(traindataset)

    # Save
    model.save(quantized_model_id)
    
    # Cleanup CPU model instance
    del model
    gc.collect()

    # ---------------------------------------------------------
    # 3. Calculate Quantized PPL
    # ---------------------------------------------------------
    print(f"--- Loading Quantized Model: {quantized_model_id} ---")
    
    # Load quantized model to GPU
    model = AutoNanoModel.load(quantized_model_id, device=device)

    # Inference check
    prompt_inputs = tokenizer("LLMs is", return_tensors="pt").to(device)
    generated = model.generate(**prompt_inputs)[0]
    print("Sample Generation:", tokenizer.decode(generated))

    # Calculate PPL
    quant_ppl = calculate_avg_ppl(model, tokenizer)
    print(f"Quantized Model PPL: {quant_ppl}")
    save_ppl_to_csv(quantized_model_id, quant_ppl, csv_output_file)

    print(f"Comparison complete. Results saved to {csv_output_file}")


if __name__ == "__main__":
    main()