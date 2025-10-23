import logging
import os
from typing import Dict, List

from transformers import AutoTokenizer

from nanomodel import AutoNanoModel, get_best_device

# Environment configuration for predictable CUDA ordering and expandable allocator
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

PRETRAINED_MODEL_ID = "Qwen/Qwen3-1.7B"
QUANTIZED_MODEL_ID = "qwen3-1.7b-gptq-4bit"


def build_chat_prompt(tokenizer: AutoTokenizer, messages: List[Dict[str, str]]) -> str:
    """
    Render the conversation into a single prompt. Prefer the tokenizer's chat template
    when available to keep model-specific system formatting intact.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Fallback prompt if the tokenizer does not ship a chat template.
    conversation = []
    for message in messages:
        role = message["role"].capitalize()
        conversation.append(f"{role}: {message['content']}")
    conversation.append("Assistant:")
    return "\n".join(conversation)


def chat():
    logging.getLogger("NanoModel").info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_ID, use_fast=True)

    device = get_best_device()
    logging.getLogger("NanoModel").info("Loading quantized model...")
    model = AutoNanoModel.load(QUANTIZED_MODEL_ID, device=device)

    chat_history: List[Dict[str, str]] = []
    print("Interactive chat is ready. Type 'exit' or 'quit' to stop.")

    while True:
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        chat_history.append({"role": "user", "content": user_input})

        prompt = build_chat_prompt(tokenizer, chat_history)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        generate_kwargs = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
        }

        output_ids = model.generate(**inputs, **generate_kwargs)
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        assistant_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if not assistant_text:
            assistant_text = "[No response generated]"

        chat_history.append({"role": "assistant", "content": assistant_text})
        print(f"Assistant: {assistant_text}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    chat()
