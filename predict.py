from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = "./models"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

text = "pinyin2text: wasilafe·fumiqi·nijinsiji（；；，）shibolanyieluusbbaleiwuzhehebiaobujiawyifeifakdewudaojiqiaojbduijiaosekehuadeshenduerwenmingcb"

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

result = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print(result)
