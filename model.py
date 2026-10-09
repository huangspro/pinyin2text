from transformers import AutoTokenizer, MT5ForConditionalGeneration, T5ForConditionalGeneration, Seq2SeqTrainingArguments, AutoConfig, Seq2SeqTrainer
import tool
from datasets import Dataset
import sys, os
from rich.console import Console
from rich.panel import Panel

console = Console()
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

dataset_path = "./small_dataset"
model_name = 'google/mt5-base'
# check_point = "checkpoints/checkpoint-4220"
model_path = './models'
# model_path = './added_word_clear_model'
# model_path = './added_word_pretrained_model'

# set training mode: 
# 1->train from official pretrained model
# 2->train from scratch, but official model configure
# 3->train from a trained model
# 4->train from a check point
TRAIN_MODE = 1






# load model
console.print(Panel(
    "loading " + model_name,
    title="loading tokenizers and models",
    border_style="cyan"
))
if TRAIN_MODE == 1:
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
    model = MT5ForConditionalGeneration.from_pretrained(model_name, local_files_only=False)
elif TRAIN_MODE == 2:
    # load config, do not load model
    config = AutoConfig.from_pretrained(model_name, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = MT5ForConditionalGeneration(config)
elif TRAIN_MODE == 3:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = MT5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
elif TRAIN_MODE == 4:
    tokenizer = AutoTokenizer.from_pretrained(check_point, local_files_only=True)
    model = MT5ForConditionalGeneration.from_pretrained(check_point, local_files_only=True)
    
    
    
    
# load my dataset from list
console.print(Panel(
    "loading ",
    title="loading dataset from list",
    border_style="cyan"
))
my_dataset = Dataset.from_list(tool.load_my_dataset(1, dataset_path))


# process dataset
MAX_LEN = 128

def tokenize_function(examples):
    model_inputs = tokenizer(
        examples["input"], padding="max_length", truncation=True, max_length=MAX_LEN
    )
    labels = tokenizer(
        text_target=examples["output"], padding="max_length",
        truncation=True, max_length=MAX_LEN
    )
    # deal with the padding problem
    label_ids = [
        [(l if l != tokenizer.pad_token_id else -100) for l in seq]
        for seq in labels["input_ids"]
    ]
    model_inputs["labels"] = label_ids
    return model_inputs

# split dataset into train set and eval set
split_dataset = my_dataset.train_test_split(test_size=0.1, seed=42)
tokenized_train = split_dataset["train"].map(tokenize_function, batched=True, remove_columns=split_dataset["train"].column_names)
tokenized_eval  = split_dataset["test"].map(tokenize_function, batched=True, remove_columns=split_dataset["test"].column_names)
print("-"*10, "dataset ready", ", train_set length: ", len(tokenized_train), "-"*10)







# set training super arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./checkpoints",
    eval_strategy="epoch",          
    save_strategy="epoch",
    learning_rate=2e-6,
    per_device_train_batch_size=2,  # batch size
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    weight_decay=0,
    save_total_limit=3,
    predict_with_generate=True,
    fp16=True,                       
    logging_steps=50,
)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
)







if TRAIN_MODE == 4:
    trainer.train(resume_from_checkpoint=check_point)
else:
    trainer.train()
print("finish training, model file saved to ", model_path)
trainer.save_model(model_path)  # save the model checkpoint
tokenizer.save_pretrained(model_path)  # save the tokenizer
