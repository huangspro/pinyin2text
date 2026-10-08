from pypinyin import lazy_pinyin
import random
import os

# get the pinyin label of a sentence, add noise and so on
def get_pinyin(text = "",
        add_noise = True,
        noise_intensity = 0,
        remove_punctuation = False,
        random_remove = False,
        
        keep_space = False
        ):
        
    tem = list("qwertyuioplkmjnhbgvfcdxsza")  # the letters used to replace
    result = lazy_pinyin(text)  # get the pinyin result 
    probability = 0.01 # probablity that the add_noise and random_remove
    
    # randomly replace some letters
    if add_noise == True:
        for ii in range(noise_intensity):
            for i in range(0, len(result)):
                if random.randint(1, 100) <= 100*probability:
                    index = random.randint(0,len(result[i])-1)
                    result[i] = result[i][0:index] + tem[random.randint(0,len(tem)-1)] + result[i][index+1:len(result[i])]
              
    # randomly remove some letters
    if random_remove == True:
        for i in range(0, len(result)):
            if random.randint(1, 100) <= 100*probability:
                index = random.randint(0,len(result[i])-1)
                result[i] = result[i][0:index] + result[i][index+1:len(result[i])]
            
    # remove punctuation       
    if remove_punctuation == True:
        result = [i for i in result if i not in ["，", "。", "“", "”", "：", "？", "！", "《", "》", "『", "』"]]        
    
    return (' ' if keep_space else '').join(result).replace("\n", "")
            

# load text from a file into a list
def get_text(file_path):
    result = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i in f:
            result.append(i)
    return result
    

"""
[
    {"input": "pinyin2text: nihao", "output": "你好"}
]
"""
def load_my_dataset(number_of_files):
    data = []
    dire = "./wiki2019zh_corpus"
    count = number_of_files  # control the number of files
    
    for i in os.listdir(dire):
        if count>0:
            text = get_text(dire + "/" + i)  # return a list of sentences
            pinyin_text = [get_pinyin(x, True, 1, False if random.randint(1,100)>=50 else True, False if random.randint(1,100)>=50 else True, False if random.randint(1,100)>=50 else True) for x in text if x != ""]  # transform text into pinyin list
            #pinyin_text = [get_pinyin(x, False, 0, False, False, False) for x in text if x != ""]  # transform text into pinyin list
            count -= 1
        else:
            break
    
    result = []
    for i in range(len(text)):
        result.append({"input": "pinyin2text: " + pinyin_text[i], "output": text[i].replace("\n", "")})
    return result
    
