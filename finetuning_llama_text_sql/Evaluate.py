import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, pipeline

print('Imports Resolved')

peft_model_id = "./finetune-llama-7b-text-to-sql"
# peft_model_id = args.output_dir
 
# Load Model with PEFT adapter
model = AutoPeftModelForCausalLM.from_pretrained(
  peft_model_id,
  device_map="auto",
  torch_dtype=torch.float16
)

print('Model Download Complete')

tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

print('Tokenizer Set')
# load into pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

print('Pipeline Formed')

from datasets import load_dataset
from random import randint

# Load our test dataset
eval_dataset = load_dataset("json", data_files="test_dataset.json", split="train")
rand_idx = randint(0, len(eval_dataset))


print('Evaluation Dataset Ready')


from tqdm import tqdm
 
 
def evaluate(sample):
    prompt = pipe.tokenizer.apply_chat_template(sample["messages"][:2], tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
    predicted_answer = outputs[0]['generated_text'][len(prompt):].strip()
    if predicted_answer == sample["messages"][2]["content"]:
        return 1
    else:
        return 0
 
success_rate = 0
number_of_eval_samples = 250
eval_dataset = eval_dataset.shuffle().select(range(number_of_eval_samples))

# iterate over eval dataset and predict
for i,s in enumerate(eval_dataset):
    success_rate += evaluate(s)
    accuracy = success_rate / (i+1)
    print(f" Evaluation Progress: {i}/{number_of_eval_samples}. Accuracy: {accuracy*100:.2f}%")
 
# compute accuracy
accuracy = success_rate / number_of_eval_samples

print('\n\n')
print('-'*30)
print(f"Overall Accuracy: {accuracy*100:.2f}%")