### datasets,Hugging Face

- list_datasets
  > > > from datasets import list_datasets
  > > > all_datasets = list_datasets()
  > > > <stdin>:1: FutureWarning: list_datasets is deprecated and will be removed in the next major version of datasets. Use 'huggingface_hub.list_datasets' instead.

from transformer import
AutoModelForCausalLLM,
AutoTokenizer,
pipeline

model = AutoModelForCausalLLM.from_pretrained()
tokenizer = AutoTokenizer.from_pretrained()

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened.<|assistant|>"

input_ids = tokenizer(prompt,return_tensors="pt").input_ids.to("cpu")

generation_output = model.generate(
input_ids=input_ids,
max_new_tokens=20
)
