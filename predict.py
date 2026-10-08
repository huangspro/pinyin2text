from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = "./added_word_clear_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

print("loaded") 
while True:
    text = "pinyin2text: " + input("")
    print("input: ", text)
    if text == "pinyin2text: exit":
        break
    inputs = tokenizer(
        text,
        max_length=128,
        truncation=True,
        return_tensors="pt"
    )

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=128,
    )
    print(outputs) 
    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    print(result)
