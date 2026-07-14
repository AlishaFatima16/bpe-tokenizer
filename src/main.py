"""
main.py — Command-line interface for the BPE Tokenizer.

Usage examples:
    python -m src.main train --corpus data/corpus.txt --vocab-size 1000
    python -m src.main encode --text "hello world"
    python -m src.main decode --ids 104 101 108 108 111
    python -m src.main analyse --text "the quick brown fox"
    python -m src.main interactive
"""

import argparse
import sys
import os

from src.tokenizer import BPETokenizer
from src.trainer import train_and_save
from src.utils import visualize_tokens, vocab_stats, compare_texts

DEFAULT_MODEL = "models/tokenizer.json"


# -------------------------------------------------------------------
# Command handlers
# -------------------------------------------------------------------

def cmd_train(args):
    """Train a new BPE tokenizer on a corpus file."""
    if not os.path.exists(args.corpus):
        print(f"Error: corpus file not found: {args.corpus}")
        sys.exit(1)

    train_and_save(
        corpus_path=args.corpus,
        save_path=args.model,
        vocab_size=args.vocab_size,
        verbose=args.verbose,
        max_chars=args.max_chars,
    )


def cmd_encode(args):
    """Encode text into token IDs."""
    tok = _load_model(args.model)
    ids = tok.encode(args.text)

    print(f"\nInput : '{args.text}'")
    print(f"Tokens: {len(ids)}")
    print(f"IDs   : {ids}")

    if args.verbose:
        print("\nBreakdown:")
        for i, token_id in enumerate(ids):
            token_bytes = tok.vocab[token_id]
            try:
                token_str = token_bytes.decode("utf-8")
            except UnicodeDecodeError:
                token_str = repr(token_bytes)
            print(f"  [{i:>3}] id={token_id:>5}  '{token_str}'")


def cmd_decode(args):
    """Decode token IDs back into text."""
    tok = _load_model(args.model)
    ids = [int(x) for x in args.ids]
    text = tok.decode(ids)

    print(f"\nIDs  : {ids}")
    print(f"Text : '{text}'")


def cmd_analyse(args):
    """Analyse how a text is tokenized with full visualisation."""
    tok = _load_model(args.model)

    vocab_stats(tok)
    visualize_tokens(tok, args.text)

    ratio = tok.compression_ratio(args.text)
    print(f"\nCompression ratio: {ratio:.2f}x")
    print(f"({len(args.text.encode('utf-8'))} bytes → {len(tok.encode(args.text))} tokens)")


def cmd_compare(args):
    """Compare tokenization across multiple texts."""
    tok = _load_model(args.model)
    texts = args.texts
    compare_texts(tok, texts)


def cmd_interactive(args):
    """Interactive mode — type text, see tokens in real time."""
    tok = _load_model(args.model)

    print("\n" + "=" * 55)
    print("  BPE Tokenizer — Interactive Mode")
    print("  Type text to tokenize. Commands: :quit :stats :help")
    print("=" * 55 + "\n")

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        elif user_input == ":quit":
            print("Bye!")
            break
        elif user_input == ":stats":
            vocab_stats(tok)
        elif user_input == ":help":
            print("  :stats  — show vocabulary statistics")
            print("  :quit   — exit")
            print("  Any other input is tokenized and displayed")
        else:
            ids = tok.encode(user_input)
            ratio = tok.compression_ratio(user_input)
            print(f"  Tokens : {len(ids)}  |  Compression: {ratio:.2f}x")
            print(f"  IDs    : {ids}")

            # Show each token
            parts = []
            for token_id in ids:
                token_bytes = tok.vocab[token_id]
                try:
                    parts.append(f"[{token_bytes.decode('utf-8')}]")
                except UnicodeDecodeError:
                    parts.append(f"[{repr(token_bytes)}]")
            print(f"  Split  : {''.join(parts)}")
            print()


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _load_model(path: str) -> BPETokenizer:
    if not os.path.exists(path):
        print(f"Error: model not found at '{path}'")
        print("Train one first:  python -m src.main train --corpus data/corpus.txt")
        sys.exit(1)
    tok = BPETokenizer()
    tok.load(path)
    return tok


# -------------------------------------------------------------------
# Argument parser
# -------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bpe-tokenizer",
        description="BPE Tokenizer — built from scratch",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Path to saved tokenizer (default: {DEFAULT_MODEL})",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Train a new tokenizer")
    p_train.add_argument("--corpus", default="data/corpus.txt")
    p_train.add_argument("--vocab-size", type=int, default=1000)
    p_train.add_argument("--max-chars", type=int, default=200_000)
    p_train.add_argument("--verbose", action="store_true")

    # encode
    p_encode = sub.add_parser("encode", help="Encode text to token IDs")
    p_encode.add_argument("--text", required=True)
    p_encode.add_argument("--verbose", action="store_true")

    # decode
    p_decode = sub.add_parser("decode", help="Decode token IDs to text")
    p_decode.add_argument("--ids", nargs="+", required=True)

    # analyse
    p_analyse = sub.add_parser("analyse", help="Visualise tokenization of text")
    p_analyse.add_argument("--text", required=True)

    # compare
    p_compare = sub.add_parser("compare", help="Compare tokenization of multiple texts")
    p_compare.add_argument("--texts", nargs="+", required=True)

    # interactive
    sub.add_parser("interactive", help="Interactive tokenizer session")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "train"      : cmd_train,
        "encode"     : cmd_encode,
        "decode"     : cmd_decode,
        "analyse"    : cmd_analyse,
        "compare"    : cmd_compare,
        "interactive": cmd_interactive,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()