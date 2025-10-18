from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

dataset = ["nanomodel is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]

gptq_config = GPTQConfig(bits=4, dataset=dataset, tokenizer=tokenizer)

quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", quantization_config=gptq_config)
quantized_model.save_pretrained("./qwen3-0.6B-gptq")

tokenizer.save_pretrained("./qwen3-0.6B-gptq")

model = AutoModelForCausalLM.from_pretrained("./qwen3-0.6B-gptq", device_map="auto")
print(tokenizer.decode(model.generate(**tokenizer("nanomodel is", return_tensors="pt").to(model.device))[0]))
