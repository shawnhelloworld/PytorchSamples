import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  
model = GPT2LMHeadModel.from_pretrained("gpt2", torchscript=True).to(device).eval() 
    

in_text = "Manchester United Football Club is"
in_tokens = torch.tensor(tokenizer.encode(in_text)).to(device)

# inference
token_eos = torch.tensor([198]).to(device)# line break symbol
out_token = None
i = 0
with torch.no_grad():
    while out_token != token_eos:
        logits, _ = model(in_tokens)
        out_token = torch.argmax(logits[-1, :], dim=0, keepdim=True)
        in_tokens = torch.cat((in_tokens, out_token), 0)
        text = tokenizer.decode(in_tokens)
        print(f'step {i} input: {text}', flush=True)
        i += 1

out_text = tokenizer.decode(in_tokens)
print(f' Input: {in_text}')
print(f'Output: {out_text}')
