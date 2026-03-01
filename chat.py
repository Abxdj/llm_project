import torch
from transformers import AutoTokenizer
from model import GPTLanguageModel  # your extracted model file
import pickle
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# SELECT MODEL TYPE

MODE = "token"   # change to "char" if needed

# LOAD MODEL

if MODE == "token":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=256,
        n_head=8,
        n_layer=4,
        block_size=256,
        dropout=0.15
    )

    ckpt = torch.load("training_and_notebook_stuff/token_gpt_best_model_256.pt", map_location=device)

elif MODE == "char":
    BASE_DIR = os.path.dirname(__file__)
    VOCAB_PATH = os.path.join(BASE_DIR, "training_and_notebook_stuff", "char_vocab.pkl")

    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    string_to_int = vocab["string_to_int"]
    int_to_string = vocab["int_to_string"]

    vocab_size = len(string_to_int)
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=384,
        n_head=8,
        n_layer=8,
        block_size=128,
        dropout=0.15
    )

    ckpt_path = os.path.join(BASE_DIR, "training_and_notebook_stuff", "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

model.load_state_dict(ckpt["model"])
model.to(device)
model.eval()

print("Model loaded")

# CHAT LOOP

temperature = 0.65
top_k = 50
max_new_tokens = 150

context = ""

while True:
    user_input = input("\nYou: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    context += "User: " + user_input + " Model: "

    if MODE == "token":
        input_ids = tokenizer.encode(context, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )

        generated_text = tokenizer.decode(output[0])
        reply = generated_text[len(context):]

    elif MODE == "char":
        encoded = torch.tensor(
            [string_to_int[c] for c in context],
            dtype=torch.long
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model.generate(
                encoded,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )

        decoded = ''.join([int_to_string[i.item()] for i in output[0]])
        reply = decoded[len(context):]

    print("Model:", reply.strip())

    context += reply