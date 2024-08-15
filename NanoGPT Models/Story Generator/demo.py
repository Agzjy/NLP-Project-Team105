# Demo Notebook

import torch
import sentencepiece as spm
from load_model import GPTLanguageModel

# Load SentencePiece model
sp = spm.SentencePieceProcessor(model_file='spm_model.model')

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the GPT model
model = GPTLanguageModel().to(device)
model.eval()  # Set the model to evaluation mode

# Load the model checkpoint
checkpoint_path = 'model_checkpoint.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Encode and decode functions using SentencePiece
def encode(text):
    return torch.tensor(sp.encode(text, out_type=int), dtype=torch.long, device=device).unsqueeze(0)

def decode(ids):
    return sp.decode(ids.squeeze().tolist())

# Function to generate text from a prompt
def generate_text(prompt, max_new_tokens=100):
    # Encode the prompt
    input_ids = encode(prompt)
    
    # Generate text
    generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
    
    # Decode the generated IDs to text
    generated_text = decode(generated_ids)
    
    return generated_text

# Demo Interface
prompt = "Once upon a time"
print(f"Prompt: {prompt}")
generated_text = generate_text(prompt, max_new_tokens=100)
print("\nGenerated text:\n")
print(generated_text)

prompt = "Good doctor"
print(f"Prompt: {prompt}")
generated_text = generate_text(prompt, max_new_tokens=100)
print("\nGenerated text:\n")
print(generated_text)

prompt = "Bad day"
print(f"Prompt: {prompt}")
generated_text = generate_text(prompt, max_new_tokens=100)
print("\nGenerated text:\n")
print(generated_text)