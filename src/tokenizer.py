"""
BPE (Byte Pair Encoding) Tokenizer — built from scratch.

Core idea:
  1. Represent text as raw UTF-8 bytes (256 base tokens)
  2. Find the most frequent adjacent pair of tokens
  3. Merge it into a new single token
  4. Repeat until vocabulary size is reached

Training produces:  merges  (the ordered merge rules)
                    vocab   (id → bytes mapping)
"""


# ---------------------------------------------------------------------------
# Low-level helpers — these two functions are the entire BPE algorithm
# ---------------------------------------------------------------------------

def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """
    Count how often every adjacent pair appears in the token sequence.

    Example:
        ids = [1, 2, 3, 1, 2]
        → {(1,2): 2, (2,3): 1, (3,1): 1}
    """
    counts = {}
    for pair in zip(ids, ids[1:]):          # slide a window of size 2
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """
    Replace every occurrence of `pair` in `ids` with `new_id`.

    Example:
        ids    = [1, 2, 3, 1, 2]
        pair   = (1, 2)
        new_id = 99
        → [99, 3, 99]
    """
    result = []
    i = 0
    while i < len(ids):
        # If current + next match the pair, merge them
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            result.append(new_id)
            i += 2          # skip both tokens we just merged
        else:
            result.append(ids[i])
            i += 1
    return result


# ---------------------------------------------------------------------------
# BPE Tokenizer class
# ---------------------------------------------------------------------------

class BPETokenizer:
    """
    Byte Pair Encoding tokenizer trained on raw text.

    After training:
        self.merges  — ordered list of merges: {(a, b): new_id}
        self.vocab   — id → bytes:             {id: bytes}
    """

    def __init__(self):
        # merges are learned during training
        self.merges: dict[tuple[int, int], int] = {}

        # vocab maps token id → the actual bytes it represents
        # start with the 256 raw byte tokens
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # ------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------

    def train(self, text: str, vocab_size: int, verbose: bool = True) -> None:
        """
        Learn BPE merge rules from `text` until vocabulary reaches `vocab_size`.

        Args:
            text:       Raw training text (any language, any content)
            vocab_size: Target vocabulary size. Must be >= 256.
            verbose:    Print progress of each merge.
        """
        assert vocab_size >= 256, "vocab_size must be at least 256 (one per byte)"

        num_merges = vocab_size - 256       # how many merge steps we need

        # Step 1: convert text → UTF-8 bytes → list of ints (0-255)
        tokens = list(text.encode("utf-8"))

        print(f"Training BPE tokenizer...")
        print(f"  Text length   : {len(text):,} characters")
        print(f"  Tokens (bytes): {len(tokens):,}")
        print(f"  Base vocab    : 256 bytes")
        print(f"  Target vocab  : {vocab_size}")
        print(f"  Merges needed : {num_merges}")
        print()

        # Step 2: repeatedly find the best pair and merge it
        for i in range(num_merges):
            # count all adjacent pairs
            stats = get_stats(tokens)

            if not stats:
                print("No more pairs to merge. Stopping early.")
                break

            # pick the pair with the highest count
            best_pair = max(stats, key=lambda p: stats[p])
            best_count = stats[best_pair]

            # assign it the next available token id
            new_id = 256 + i

            # apply the merge across the entire token sequence
            tokens = merge(tokens, best_pair, new_id)

            # save the merge rule
            self.merges[best_pair] = new_id

            # update vocab: new token = concatenation of the two it replaced
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            if verbose:
                merged_str = self.vocab[new_id]
                print(
                    f"  Merge {i+1:>4}/{num_merges}: "
                    f"({best_pair[0]:>3}, {best_pair[1]:>3}) → {new_id} "
                    f"| freq={best_count:>6} "
                    f"| '{self._safe_repr(merged_str)}'"
                )

        print(f"\nTraining complete. Vocabulary size: {len(self.vocab)}")

    # ------------------------------------------------------------------
    # ENCODE
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """
        Convert a string into a list of token IDs using learned merge rules.
        """
        # start from raw bytes
        tokens = list(text.encode("utf-8"))

        # apply merges in the exact order they were learned
        for pair, new_id in self.merges.items():
            tokens = merge(tokens, pair, new_id)

        return tokens

    # ------------------------------------------------------------------
    # DECODE
    # ------------------------------------------------------------------

    def decode(self, ids: list[int]) -> str:
        """
        Convert a list of token IDs back into a string.
        """
        # look up bytes for each token id, concatenate
        byte_seq = b"".join(self.vocab[i] for i in ids)

        # decode bytes → string, replacing any invalid UTF-8 sequences
        return byte_seq.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # SAVE / LOAD
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save merge rules and vocab to a file."""
        import json

        data = {
            "merges": {f"{a},{b}": new_id for (a, b), new_id in self.merges.items()},
            "vocab": {str(k): list(v) for k, v in self.vocab.items()},
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"Tokenizer saved to: {path}")

    def load(self, path: str) -> None:
        """Load merge rules and vocab from a saved file."""
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.merges = {
            tuple(int(x) for x in k.split(",")): v
            for k, v in data["merges"].items()
        }
        self.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}

        print(f"Tokenizer loaded from: {path}")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _safe_repr(self, b: bytes) -> str:
        """Human-readable representation of a byte sequence."""
        try:
            return b.decode("utf-8")
        except UnicodeDecodeError:
            return repr(b)

    def compression_ratio(self, text: str) -> float:
        """
        How much did we compress? 
        Ratio = original bytes / encoded tokens
        Higher = better compression.
        """
        original = len(text.encode("utf-8"))
        encoded = len(self.encode(text))
        return original / encoded