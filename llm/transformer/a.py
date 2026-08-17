from transfromers import AutoTokenizer, AutoModelForCausalLM, pipeline


model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)


question = "What is the capital of France?"
tokenized_input = tokenizer.encode(question, return_tensors="pt")
output_tokens = model.generate(tokenized_input, max_length=50)
output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print(output_text)