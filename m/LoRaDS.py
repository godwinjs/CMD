from unsloth import FastLanguageModel

model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit"
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

# prep datasets
{"instruction": "What is the capital of France?", "output": "The capital of France is Paris."}
{"instruction": "Solve: 2 + 2", "output": "The answer is 4."}

# load ds
from datasets import load_dataset

dataset = load_dataset("json", data_files={"train": "train_data.jsonl", "test": "test_data.jsonl"})

# format DS with chatstyles
prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}
### Response:
"""
def preprocess_function(examples):
    inputs = [prompt_template.format(instruction=inst) for inst in examples["instruction"]]
    model_inputs = tokenizer(inputs, max_length=max_seq_length, truncation=True)
    return model_inputs
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Applying LoRa for fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "v_proj"],  # Fine-tune key attention layers
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
)

# train model
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
)
trainer.train()

# Evaluate
eval_results = trainer.evaluate()
print(f"Perplexity: {eval_results['perplexity']}")

# Save the model and tokenizer
model.save_pretrained("./finetuned_deepseek_r1")
tokenizer.save_pretrained("./finetuned_deepseek_r1")

# For local deployment for inference using llama.cpp on CMD
# ./llama.cpp/llama-cli \
#    --model unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf \
#    --cache-type-k q8_0 \
#    --threads 16 \
#    --prompt '<|User|>What is 1+1?<|Assistant|>' \
#    --n-gpu-layers 20 \
#    -no-cnv