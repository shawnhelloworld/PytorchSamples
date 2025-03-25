import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  
# 使用了预训练的 GPT-2 语言模型，并将其转换为 TorchScript 格式，然后加载到指定的设备上(GPU)
model = GPT2LMHeadModel.from_pretrained("gpt2", torchscript=True).to(device).eval() 
    

in_text = "Manchester United Football Club is"
# 代码是将一段文本 (in_text) 转换成 PyTorch 张量 (in_tokens)，并将其移动到指定的设备 (device) 上
in_tokens = torch.tensor(tokenizer.encode(in_text)).to(device)

# Initialize past_key_values
past_key_values = None

# Function to print GPU memory usage
if torch.cuda.is_available():
    def print_gpu_memory_usage():
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        print(f"GPU Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
else:
    def print_gpu_memory_usage():
        print("GPU not available.")

# inference
token_eos = torch.tensor([198]).to(device)# line break symbol
out_token = None
i = 0
out_text = in_text
with torch.no_grad():
    while out_token != token_eos:
        # Pass past_key_values to the model
        logits, past_key_values = model(in_tokens, past_key_values=past_key_values)[:2]
        out_token = torch.argmax(logits[-1, :], dim=0, keepdim=True)
        in_tokens = out_token
        text = tokenizer.decode(in_tokens)
        print(f'step {i} input: {text}', flush=True)
        i += 1
        out_text += text
        print_gpu_memory_usage()  # Print GPU memory usage
print(f' Input: {in_text}')
print(f'Output: {out_text}')

# Print final GPU memory usage
print_gpu_memory_usage()
