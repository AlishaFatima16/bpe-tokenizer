<div align="center">

# ██████╗ ██████╗ ███████╗    ████████╗ ██████╗ ██╗  ██╗███████╗███╗   ██╗██╗███████╗███████╗██████╗
# ██╔══██╗██╔══██╗██╔════╝    ╚══██╔══╝██╔═══██╗██║ ██╔╝██╔════╝████╗  ██║██║╚══███╔╝██╔════╝██╔══██╗
# ██████╔╝██████╔╝█████╗         ██║   ██║   ██║█████╔╝ █████╗  ██╔██╗ ██║██║  ███╔╝ █████╗  ██████╔╝
# ██╔══██╗██╔═══╝ ██╔══╝         ██║   ██║   ██║██╔═██╗ ██╔══╝  ██║╚██╗██║██║ ███╔╝  ██╔══╝  ██╔══██╗
# ██████╔╝██║     ███████╗       ██║   ╚██████╔╝██║  ██╗███████╗██║ ╚████║██║███████╗███████╗██║  ██║
# ╚═════╝ ╚═╝     ╚══════╝       ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝╚═╝╚══════╝╚══════╝╚═╝  ╚═╝

### **The algorithm powering GPT-4 — rebuilt from 30 lines of pure Python**

<br/>

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-17%20Passing-22c55e?style=for-the-badge&logo=pytest&logoColor=white)](#testing)
[![Compression](https://img.shields.io/badge/Compression-3.12x-f59e0b?style=for-the-badge&logo=speedtest&logoColor=white)](#results)
[![License](https://img.shields.io/badge/License-MIT-8b5cf6?style=for-the-badge)](LICENSE)

<br/>

> **"I wanted to understand tokenization at the atomic level.**
>
> **Not call a library. Not copy a tutorial. Build it — byte by byte."**

</div>

---

# ⚡ What Is This?

This is a **ground-up implementation of Byte Pair Encoding (BPE)** — the tokenization algorithm at the heart of **GPT-2, GPT-4, LLaMA, Claude, and virtually every modern large language model.**

No `transformers`.

No `tiktoken`.

No shortcuts.

Just Python, a frequency counter, and a merge function — the same two primitives that **Sennrich et al.** described in their 2015 paper that changed NLP forever.

---

# 🧠 The Idea in 60 Seconds

Language models don't read text.

They read **numbers**.

Tokenization is the bridge.

```text
"tokenization"
        │
        ▼
      BPE
        │
        ▼
[529, 107, 270, 864, 846]
```

But how do you decide what gets a number?

That's the whole problem.

### **BPE's answer is beautifully simple.**

### Step 1

Start with **256 tokens**—one for every UTF-8 byte.

Every possible input is already covered.

```text
h   e   l   l   o       w   o   r   l   d

104 101 108 108 111 32 119 111 114 108 100
```

---

### Step 2

Count the most frequent adjacent pair across the training corpus.

```text
('l','o') appears 312 times

↓

Create token 257
```

---

### Step 3

Merge it.

Repeat hundreds of times.

```text
Common words
↓

Single Tokens

Rare words
↓

Known subword pieces
```

Result?

- ✅ Zero unknown tokens
- ✅ Better compression
- ✅ Faster language modelling

That's it.

Two functions.

Infinite depth.

---

# 🏗️ Architecture

```text
bpe-tokenizer/
│
├── src/
│   ├── tokenizer.py      ← Core BPE engine
│   ├── trainer.py        ← Training pipeline
│   ├── utils.py          ← Visualisation & compression stats
│   └── main.py           ← CLI
│
├── data/
│   └── corpus.txt        ← Shakespeare corpus
│
├── models/
│   └── tokenizer.json    ← Saved vocabulary + merges
│
├── tests/
│   └── test_tokenizer.py ← 17 unit tests
│
└── notebooks/
    └── explore.ipynb     ← Interactive walkthrough
```

---

# 🔬 The Core — 30 Lines That Do Everything

```python
def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """
    Count how often every adjacent pair appears.

    This single function drives the entire BPE algorithm.

    Example:

    [104,101,108,108,111]

↓

{
 (104,101):1,
 (101,108):1,
 (108,108):1,
 (108,111):1
}
    """

    counts = {}

    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1

    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """
    Replace every occurrence of `pair`
    with `new_id`.

    Non-overlapping.
    Left-to-right.

    Example:

    [1,2,3,1,2]

+

pair=(1,2)

↓

[99,3,99]
    """

    result = []
    i = 0

    while i < len(ids):

        if (
            i < len(ids) - 1
            and ids[i] == pair[0]
            and ids[i + 1] == pair[1]
        ):
            result.append(new_id)
            i += 2

        else:
            result.append(ids[i])
            i += 1

    return result
```

Everything else—

- the `BPETokenizer` class,
- the trainer,
- the CLI,
- saving/loading,
- visualization—

is scaffolding around these two functions.

---

# 🚀 Quick Start

```bash
# Clone repository

git clone https://github.com/AlishaFatima16/bpe-tokenizer.git

cd bpe-tokenizer

# Install dependencies

pip install -r requirements.txt

# Train tokenizer

python -m src.main train \
--corpus data/corpus.txt \
--vocab-size 1000

# Encode text

python -m src.main encode \
--text "the organization tokenizer" \
--verbose

# Analyse tokenization

python -m src.main analyse \
--text "machine learning and tokenization"

# Interactive shell
---

# 🎯 Live Output

Encoding:

```text
"the organization tokenizer"
```

Output:

```text
Input  : 'the organization tokenizer'

Tokens : 14

IDs:
[307, 270, 103, 264, 105, 122, 595, 403, 364, 107, 268, 105, 122, 263]
```

Token breakdown:

```text
[ 0]  'the '    ← single token — so frequent it earned one

[ 1]  'or'

[ 2]  'g'

[ 3]  'an'

[ 4]  'i'

[ 5]  'z'

[ 6]  'ati'     ← subword pattern discovered from data

[ 7]  'on '

[ 8]  'to'

[ 9]  'k'

[10]  'en'

[11]  'i'

[12]  'z'

[13]  'er'
```

---

## Watching BPE Learn

The tokenizer discovers patterns automatically:

```text
Merge   1/744:
(101, 32) → 256

Frequency:
4,891

Pattern:
'e '


Merge   2/744:
(116,104) → 257

Frequency:
4,201

Pattern:
'th'


Merge   3/744:
(257,101) → 258

Frequency:
4,198

Pattern:
'the'


...


Merge  39/744:
(108,263) → 294

Frequency:
312

Pattern:
'lan'


Merge  40/744:
(294,103) → 295

Frequency:
312

Pattern:
'lang'


Merge  41/744:
(295,117) → 296

Frequency:
312

Pattern:
'langu'


Merge  42/744:
(296,97) → 297

Frequency:
312

Pattern:
'langua'


Merge  43/744:
(297,103) → 298

Frequency:
312

Pattern:
'languag'


Merge  44/744:
(298,256) → 299

Frequency:
312

Pattern:
'language '
```

> **Nobody programmed the word `language`.**
>
> **It emerged from frequency alone.**

---

# 📊 Results

Trained on **200,000 characters** of the Tiny Shakespeare dataset.

| Metric | This Implementation | GPT-2 |
|---|---:|---:|
| Vocabulary size | 1,000 | 50,257 |
| Training data | 200K chars | ~40GB |
| Compression ratio | **3.12×** | ~3.5× |
| Average token length | 3.09 bytes | ~4 bytes |
| Unknown tokens | 0 | 0 |
| Tests passing | **17 / 17** | — |
| Training time | 77s (Python) | Hours (Rust) |

---

Achieving **3.12× compression** with only **1,000 tokens** compared to GPT-2's approximately **3.5× compression** with **50,257 vocabulary entries**.

This implementation achieves this using:

- Smaller vocabulary
- Tiny training dataset
- Pure Python implementation
- No external tokenizer libraries

---

# ✅ Testing

Run:

```bash
pytest tests/ -v
```

Results:

```text
tests/test_tokenizer.py::TestGetStats::test_basic_pair_count          PASSED
tests/test_tokenizer.py::TestGetStats::test_single_token              PASSED
tests/test_tokenizer.py::TestGetStats::test_empty                     PASSED
tests/test_tokenizer.py::TestGetStats::test_all_same                  PASSED

tests/test_tokenizer.py::TestMerge::test_basic_merge                  PASSED
tests/test_tokenizer.py::TestMerge::test_no_match                     PASSED
tests/test_tokenizer.py::TestMerge::test_overlapping_pairs            PASSED
tests/test_tokenizer.py::TestMerge::test_empty                        PASSED

tests/test_tokenizer.py::TestBPETokenizer::test_vocab_size             PASSED
tests/test_tokenizer.py::TestBPETokenizer::test_merge_count            PASSED
tests/test_tokenizer.py::TestBPETokenizer::test_encode_returns_ints    PASSED
tests/test_tokenizer.py::TestBPETokenizer::test_decode_inverse         PASSED
tests/test_tokenizer.py::TestBPETokenizer::test_compression             PASSED
tests/test_tokenizer.py::TestBPETokenizer::test_empty_string            PASSED
tests/test_tokenizer.py::TestBPETokenizer::test_unicode_text            PASSED
tests/test_tokenizer.py::TestBPETokenizer::test_base_vocab              PASSED

tests/test_tokenizer.py::TestSaveLoad::test_save_load_roundtrip        PASSED
```

```text
17 passed in 4.86s
```

---

# 🌍 Unicode — Every Language, Zero Configuration

Because BPE works on raw UTF-8 bytes, it supports every language without requiring language-specific rules.

Examples:

```python
tok.decode(tok.encode("café naïve résumé"))

# Output:
# café naïve résumé
```

```python
tok.decode(tok.encode("مرحبا بالعالم"))

# Output:
# مرحبا بالعالم
```

```python
tok.decode(tok.encode("你好世界 🌏"))

# Output:
# 你好世界 🌏
```

---

## Guarantee

```python
decode(encode(text)) == text
```

Always.

No information loss.

---

# 🔭 What's Next — Organisation Tokenizer

This project is **Phase A** of a larger mission.

The next step is building the **Organisation Tokenizer** for **Orgni** — a Live Organisational Intelligence Model.

The goal:

Transform every business signal into a unified intelligence stream.

```text
Emails        Documents       Databases

Calendars     APIs            IoT Events

   │              │              │
   └──────────────┴──────────────┘

              │

     Organisation Tokenizer

          (Team 1 — Me)

              │

 Unified Organisational Token Stream

              │

        O1 Model (671B params)

              │

 Next Best Action / Response / Prediction
```

---

Understanding BPE from first principles is the foundation.

The Organisation Tokenizer extends this idea beyond natural language.

It will handle:

- Structured enterprise data
- Entity markers
- Temporal signals
- Multi-modal organisational inputs
- Business-specific vocabulary

Instead of learning only language patterns, it learns **organisational semantics**.

---

# 📚 References

- 📄 **Sennrich et al. (2015)**  
  *Neural Machine Translation of Rare Words with Subword Units*

- 🎥 **Andrej Karpathy**  
  *Let's Build the GPT Tokenizer*

- 🔧 **OpenAI tiktoken**  
  Production-grade BPE tokenizer implementation

- 📖 **Tiny Shakespeare Dataset**  
  Training corpus used in this project

---

<div align="center">

## Built from scratch. Tested thoroughly. Understood completely.

⭐ **Star this repository if it helped you understand tokenization.**

</div>
## 👩‍💻 Author

**Alisha Fatima**

AI Engineering Student | Building AI systems, agentic workflows and intelligent infrastructure

GitHub: @AlishaFatima16
LinkedIn: [your link]

python -m src.main interactive
```
