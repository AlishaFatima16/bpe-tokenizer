# BPE Tokenizer — Built from Scratch

A clean, fully tested implementation of the **Byte Pair Encoding (BPE)** 
tokenization algorithm in pure Python — the same algorithm used by GPT-2, 
GPT-4, and LLaMA.

Built to deeply understand how modern language model tokenizers work at the 
lowest level, before applying the same principles to organizational data 
tokenization.

---

## What is BPE?

Tokenization converts raw text into discrete units (tokens) that a language 
model can process. BPE builds a vocabulary of subword units by:

1. Starting with **256 base tokens** — one per UTF-8 byte
2. Finding the **most frequent adjacent pair** in the training corpus
3. **Merging** that pair into a new token
4. Repeating until the target vocabulary size is reached

The result: common words become single tokens, rare words are split into 
known subword pieces. No unknown tokens — ever.

---

## Features

- **Pure Python** — zero ML framework dependencies
- **Full BPE pipeline** — train, encode, decode, save, load
- **Lossless** — decode(encode(text)) == text, always
- **Unicode-safe** — handles any language, emoji, special characters
- **CLI tool** — train and tokenize from the command line
- **17 unit tests** — all passing
- **3.12x compression** on real text (comparable to GPT-2's ~3.5x)

---

## Project Structure
bpe-tokenizer/ ├── src/ │ ├── tokenizer.py # Core BPE: get_stats(), merge(), BPETokenizer │ ├── trainer.py # Training pipeline with save/load │ ├── utils.py # Visualization and analysis helpers │ └── main.py # CLI entry point ├── data/ │ └── corpus.txt # Training corpus (Shakespeare) ├── models/ │ └── tokenizer.json # Trained tokenizer (vocab + merge rules) ├── tests/ │ └── test_tokenizer.py # 17 tests covering all components └── notebooks/ └── explore.ipynb # Interactive exploration

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train on included corpus
python -m src.main train --corpus data/corpus.txt --vocab-size 1000

# Encode text
python -m src.main encode --text "the organization tokenizer" --verbose

# Analyse tokenization
python -m src.main analyse --text "machine learning and tokenization"

# Interactive mode
python -m src.main interactive
Results
Trained on 200,000 characters of Shakespeare (tinyshakespeare):

Metric	Value
Vocabulary size	1,000 tokens
Merge rules learned	744
Avg token length	3.09 bytes
Compression ratio	3.12x
Training time	~77 seconds
Tests passing	17 / 17 ✅
GPT-2 achieves ~3.5x compression with 50,257 tokens trained on the entire internet. This implementation achieves 3.12x with 1,000 tokens on 200K characters — a strong result for the vocabulary size.

How It Works
Core Algorithm (30 lines)
def get_stats(ids):
    """Count frequency of every adjacent pair."""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts
def merge(ids, pair, new_id):
    """Replace every occurrence of pair with new_id."""
    result = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1
    return result

The entire BPE algorithm is these two functions — the rest is scaffolding.

Why I Built This
This is Phase A of a larger project: building the Organization Tokenizer for Orgni — a Live Organizational Intelligence Model that processes every signal in a business (emails, documents, databases, events) into a unified token stream for an organizational AI.

Understanding BPE from first principles is the foundation for designing a tokenizer that handles not just natural language, but structured organizational signals across multiple modalities.

References
Andrej Karpathy — Let's build the GPT Tokenizer
Neural Machine Translation of Rare Words with Subword Units — original BPE paper
OpenAI tiktoken — production BPE tokenizer
