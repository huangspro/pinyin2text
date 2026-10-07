├── added_word_pretrained_model
├── add_word_to_tokenizer.py
├── checkpoints
├── model.py
├── predict.py
├── tool.py
└── wiki2019zh_corpus

tool.py  # get pinyin representations, load dataset, and so on
model.py  # load model, dataset and tokenizer, and configure training
predict.py  # implement prediction
add_word_to_tokenizer.py  # add words to the tokenizer and save model

wiki2019zh_corpus/  # database, each file possesses about 40,000 lines of Chinese sentences. Each line contains one sentence

# The pretrained model from official hugging face hub is in /home/hhy/.cache/huggingface/hub/models--uer--t5-small-chinese-cluecorpussmall
added_word_pretrained_mode/l  # I added 26 English letters to the tokenizer and save the new model and tokenizer here

models/ The trained model checkpoints will be saved here
check_points/ The default model checkpoints save directory
