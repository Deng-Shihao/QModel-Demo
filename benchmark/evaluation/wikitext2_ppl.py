"""Quantize a model on WikiText-2 and report perplexity plus a sample decode."""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from nanomodel import AutoNanoModel, QuantizeConfig

pretrained_model_id = "Qwen/Qwen3-4B-Instruct-2507"
quantized_model_id = "/home/sd24191/git_project/QModel-Demo/quantized_models/Qwen3-4B-Instruct-2507-GPTQ-4bit"


def get_wikitext2(tokenizer, nsamples, seqlen):
    """Prepare a tokenized subset of WikiText-2 with minimum length filtering."""
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(
        lambda x: len(x["text"]) >= seqlen
    )

    return [tokenizer(example["text"]) for example in traindata.select(range(nsamples))]


@torch.inference_mode()
def calculate_avg_ppl(model, tokenizer):
    """Compute the average perplexity of the quantized model on WikiText-2."""
    from nanomodel.utils.perplexity import Perplexity

    ppl = Perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_path="wikitext",
        dataset_name="wikitext-2-raw-v1",
        split="train",
        text_column="text",
    )

    all = ppl.calculate(n_ctx=512, n_batch=512)

    # average ppl
    avg = sum(all) / len(all)

    return avg


def main():
    """Quantize on WikiText-2 data and print generation plus perplexity metrics."""
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id, use_fast=True)

    traindataset = get_wikitext2(tokenizer, nsamples=256, seqlen=1024)

    quantize_config = QuantizeConfig(
        bits=4,  # quantize model to 4-bit
        group_size=128,  # it is recommended to set the value to 128
    )

    # load un-quantized model, the model will always be force loaded into cpu
    model = AutoNanoModel.load(pretrained_model_id, quantize_config)

    # quantize model, the calibration_dataset should be list of dict whose keys can only be "input_ids" and "attention_mask"
    # with value under torch.LongTensor type.
    model.quantize(traindataset)

    # save quantized model using safetensors
    model.save(quantized_model_id)

    # load quantized model, currently only support cpu or single gpu
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoNanoModel.load(quantized_model_id, device=device)

    # inference with model.generate
    prompt_inputs = tokenizer("LLMs is", return_tensors="pt").to(device)
    generated = model.generate(**prompt_inputs)[0]
    print(tokenizer.decode(generated))

    print(
        f"Quantized Model {quantized_model_id} avg PPL is {calculate_avg_ppl(model, tokenizer)}"
    )


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
