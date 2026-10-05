from transformers import AutoTokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import tool
from datasets import Dataset

# load model
print("loading tokenizers and models")
model_name = "uer/t5-small-chinese-cluecorpussmall"
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = T5ForConditionalGeneration.from_pretrained(model_name,local_files_only=True)

# load my dataset from list
my_dataset = Dataset.from_list(tool.load_my_dataset(2))

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


training_args = Seq2SeqTrainingArguments(
    output_dir="./models",
    eval_strategy="epoch",          
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=50,
    per_device_eval_batch_size=50,
    num_train_epochs=1,
    weight_decay=0.01,
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

print("begin training")
trainer.train()
print("finish training, model file saved to ./t5-finetuned-custom-final")
trainer.save_model("./t5-finetuned-custom-final")  # save the model checkpoint
tokenizer.save_pretrained("./t5-finetuned-custom-final")  # save the tokenizer
