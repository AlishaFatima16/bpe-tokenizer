"""
trainer.py — Train a BPE tokenizer on a real text corpus and save it.
"""

import os
import time
from src.tokenizer import BPETokenizer


def load_corpus(path: str, max_chars: int = None) -> str:
    """Load text from a file, optionally capped at max_chars."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
        print(f"  (capped at {max_chars:,} characters)")
    print(f"Loaded corpus: {path}")
    print(f"  Size: {len(text):,} characters | {len(text.encode('utf-8')):,} bytes")
    return text


def train_and_save(
    corpus_path: str,
    save_path: str,
    vocab_size: int = 1000,
    verbose: bool = True,
    max_chars: int = None,        # ← add this
) -> BPETokenizer:
    start = time.time()
    text = load_corpus(corpus_path, max_chars=max_chars)   # ← pass it here
    tok = BPETokenizer()
    tok.train(text, vocab_size=vocab_size, verbose=verbose)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tok.save(save_path)
    elapsed = time.time() - start
    sample = text[:200]
    ratio = tok.compression_ratio(sample)
    print()
    print("=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    print(f"  Corpus size    : {len(text):,} characters")
    print(f"  Vocabulary size: {len(tok.vocab)}")
    print(f"  Merge rules    : {len(tok.merges)}")
    print(f"  Compression    : {ratio:.2f}x  (on first 200 chars)")
    print(f"  Training time  : {elapsed:.2f}s")
    print(f"  Saved to       : {save_path}")
    print("=" * 50)
    return tok


if __name__ == "__main__":
    train_and_save(
        corpus_path="data/corpus.txt",
        save_path="models/tokenizer.json",
        vocab_size=1000,
        verbose=False,
        max_chars=200_000,   # add this line
    )