"""Use vanilla transformers GPTQConfig for quick quantization without NanoModel."""
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

dataset = [
    (
        "nanomodel is an easy-to-use model quantization library with user-friendly"
        " APIs, based on the GPTQ algorithm."
    )
]

# Configure GPTQ via transformers; NanoModel is not required in this example.
gptq_config = GPTQConfig(bits=4, dataset=dataset, tokenizer=tokenizer)

quantized_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",
    quantization_config=gptq_config,
)
quantized_model.save_pretrained("./qwen3-0.6B-gptq")

tokenizer.save_pretrained("./qwen3-0.6B-gptq")

# Reload the quantized checkpoint with automatic device placement.
model = AutoModelForCausalLM.from_pretrained("./qwen3-0.6B-gptq", device_map="auto")
prompt_inputs = tokenizer("nanomodel is", return_tensors="pt").to(model.device)
generated = model.generate(**prompt_inputs)[0]
print(tokenizer.decode(generated))
