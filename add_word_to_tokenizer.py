from transformers import AutoTokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, AutoConfig, Seq2SeqTrainer


model_name = 'uer/t5-small-chinese-cluecorpussmall'

# 加载 tokenizer 和模型
config = AutoConfig.from_pretrained(model_name, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = T5ForConditionalGeneration(config)

# 添加自己的词
new_tokens = [i for i in "qwertyuiopasdfghjklzxcvbnm"]

num_added = tokenizer.add_tokens(new_tokens)

print("新增:", num_added)
print("当前词表大小:", len(tokenizer))

# 扩大模型 embedding
model.resize_token_embeddings(len(tokenizer))

# 保存
tokenizer.save_pretrained("./added_word_clear_model")
model.save_pretrained("./added_word_clear_model")
