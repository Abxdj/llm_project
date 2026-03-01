# 🧠 MiniGPT — Transformer Language Modeling from Scratch

This project implements a GPT-style Transformer **from scratch** in PyTorch, with two configurable modeling paradigms:

- 🧩 **Token-level GPT** using GPT-2 tokenizer  
- 🔤 **Character-level GPT** from raw text  

It includes full training scripts (via notebooks), generation logic, and a chat interface, all designed to train and evaluate locally on consumer GPU hardware (NVIDIA RTX 3060 6GB).

---

## 📌 Features

✔ Transformer implementation with:
- Multi-head self-attention  
- Positional embeddings  
- Residual connections  
- LayerNorm and Feedforward layers  
- Causal masking  

✔ Two modeling approaches:
- **Token-level** — GPT2 subword tokens, more efficient learning  
- **Character-level** — raw character prediction, educational baseline  

✔ Checkpoint saving & loading

✔ Configurable generation (temperature, top-k sampling)

✔ Interactive chat interface (`chat.py`)

✔ Clean, modular code structure

---

## 🗂 Repository Structure

```text
llm_project/
|
├── chat.py # Generation & interactive loop
├── model.py # Transformer definition
├── GPTv2.ipynb # Token GPT training notebook
├── requirements.txt
├── .gitignore
|
└── training_and_notebook_stuff/
    ├── Char_GPT.ipynb # Char GPT training notebook
    └── bigram.ipynb # Baseline bigram model
```

---

## 📥 Setup & Install

Install dependencies:

```bash
pip install -r requirements.txt
```
Dependencies include PyTorch, HuggingFace Transformers, and Datasets.

---

## 🚀 Using the Chat Interface

Run the chatbot with: python chat.py

You’ll be prompted:

You: hello
Model: ...

Use: exit or quit to terminate

You can switch between token and char modeling modes by modifying:

MODE = "token" # or "char" in chat.py.

---

## 🧪 Training Overview

- Hyperparameters are set according to my computer's specs - 16GB RAM and NVIDIA RTX 3060 Laptop GPU
---
***Token GPT (Preferred)***
- Tokenization: GPT-2 tokenizer
- Vocabulary: ~50,000 tokens
- Block size: 256
- Embedding size: 256
- Heads: 8
- Layers: 4
- Dropout: 0.15
- Dataset: Streaming WikiText-103 (approx. 8M characters)

*Structure:*
- Tokenizer encodes sequences
- Model trained autoregressively
- Loss: cross entropy
- Samples using temperature & top-k
- Token models learn structure much faster than characters.
---
***Character GPT (Baseline)***
- Tokenization: Character mapping
- Vocabulary: Unique characters in dataset
- Block size: 128
- Embedding size: 384
- Heads: 8
- Layers: 8
- Dropout: 0.15

Character models require orders of magnitude more data to produce coherent text.

---

## 🧠 Key Observations

| Model     | Coherence     | Training Efficiency | Notes                       |
| --------- | ------------- | ------------------- | --------------------------- |
| Token GPT | **Good**      | Easier              | Best performing locally     |
| Char GPT  | **Poor**      | Harder              | Only useful as baseline     |
| Bigram    | **Very weak** | Trivial             | Simple model for comparison |

Results demonstrate:
- Tokenization significantly improves learning efficiency
- Small models + limited data → limited semantic coherence
- Sampling strategies (temp & top-k) heavily influence output

---

## 🧾 Design Decisions

*Architecture*
The Transformer is implemented from first principles:
- Attention as matrix product with scaling
- Causal masks for autoregression
- LayerNorm before sublayers (Pre-Norm)
- Concatenated multi-head attention
**Equation used:**

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

---

## ⚠️ Limitations

This is a learning implementation, not a production model:
- No instruction-tuning
- Limited dataset scale
- Model sizes much smaller than GPT-2
- Not fine-tuned for conversational alignment

---

## 🔮 Future Directions

***Potential enhancements:***
- Train on larger datasets 
- Incorporate instruction fine-tuning
- Use LoRA or adapters on pretrained GPT
- Add top-p sampling
- Integrate multi-turn dialogue history

---

## 📚 Credits & References

*This project was informed and inspired by the following resources:*
- Create a Large Language Model from Scratch with Python(freeCodeCamp): https://www.youtube.com/watch?v=UU1WVnMk4E8&t=19569s
- Let's build GPT from scratch, in code, spelled out(Andrej Karpathy): https://www.youtube.com/watch?v=kCc8FmEb1nY
- FreeCodeCamp Intro to LLMs repo: https://github.com/Infatoshi/fcc-intro-to-llms
