import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

# Step 1: 加载预训练的翻译模型和对应的分词器  
model_name = "Helsinki-NLP/opus-mt-fr-en"  # 预训练的法语到英语翻译模型  
tokenizer = AutoTokenizer.from_pretrained(model_name)  
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)  

# Step 2: 定义输入的法语句子  
french_sentence = "Je suis étudiant"  # 法语输入句子  

# Step 3: 将输入句子进行分词和编码  
inputs = tokenizer(french_sentence,return_tensors="pt",padding=True,truncation=True).to(device)
# Step 4: 使用模型进行翻译推理 
output_tokens = model.generate(**inputs)
#or output_tokens = model.generate(input_ids=inputs["input_ids"]).to(device)
# Step 5:解码
outputs = tokenizer.decode(output_tokens[0],skip_special_tokens=True)
# Step6 输出翻译结果
print(f"French: {french_sentence}")  
print(f"English: {outputs}") 