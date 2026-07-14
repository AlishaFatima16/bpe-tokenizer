"""
utils.py — Visualization and analysis helpers for the BPE tokenizer.
"""

from src.tokenizer import BPETokenizer


# Assign a distinct color to each token for terminal visualization
COLORS = [
    "\033[41m", "\033[42m", "\033[43m", "\033[44m",
    "\033[45m", "\033[46m", "\033[101m", "\033[102m",
    "\033[103m", "\033[104m", "\033[105m", "\033[106m",
]
RESET = "\033[0m"


def visualize_tokens(tok: BPETokenizer, text: str) -> None:
    """
    Print the text with each token highlighted in a different color.
    Shows exactly how the tokenizer splits the input.
    """
    ids = tok.encode(text)
    print(f"\nTokenizing: '{text}'")
    print(f"Token count: {len(ids)}  |  Byte count: {len(text.encode('utf-8'))}")
    print()
    print("Colored tokens:")
    colored = ""
    for i, token_id in enumerate(ids):
        token_bytes = tok.vocab[token_id]
        try:
            token_str = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            token_str = repr(token_bytes)
        color = COLORS[i % len(COLORS)]
        colored += f"{color}{token_str}{RESET}"
    print(colored)
    print()
    print("Token breakdown:")
    for i, token_id in enumerate(ids):
        token_bytes = tok.vocab[token_id]
        try:
            token_str = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            token_str = repr(token_bytes)
        print(f"  [{i:>3}] id={token_id:>5}  '{token_str}'")


def vocab_stats(tok: BPETokenizer) -> None:
    """Print vocabulary statistics."""
    lengths = [len(v) for v in tok.vocab.values()]
    avg_len = sum(lengths) / len(lengths)
    max_token = max(tok.vocab.items(), key=lambda x: len(x[1]))
    try:
        max_str = max_token[1].decode("utf-8")
    except UnicodeDecodeError:
        max_str = repr(max_token[1])

    print("\nVocabulary Statistics")
    print("=" * 40)
    print(f"  Total tokens     : {len(tok.vocab)}")
    print(f"  Base byte tokens : 256")
    print(f"  Learned tokens   : {len(tok.merges)}")
    print(f"  Avg token length : {avg_len:.2f} bytes")
    print(f"  Longest token    : '{max_str}' ({len(max_token[1])} bytes)")
    print("=" * 40)


def compare_texts(tok: BPETokenizer, texts: list[str]) -> None:
    """
    Compare how different texts tokenize.
    Useful for understanding what the tokenizer learned.
    """
    print("\nTokenization Comparison")
    print("=" * 60)
    for text in texts:
        ids = tok.encode(text)
        ratio = len(text.encode("utf-8")) / len(ids)
        print(f"  '{text}'")
        print(f"   → {len(ids)} tokens | {ratio:.2f}x compression | ids: {ids}")
        print()