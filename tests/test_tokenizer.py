"""
Tests for the BPE tokenizer.
Run with:  pytest tests/ -v
"""

import pytest
from src.tokenizer import BPETokenizer, get_stats, merge


# ---------------------------------------------------------------
# Unit tests: core functions
# ---------------------------------------------------------------

class TestGetStats:
    def test_basic_pair_count(self):
        ids = [1, 2, 1, 2, 3]
        stats = get_stats(ids)
        assert stats[(1, 2)] == 2
        assert stats[(2, 1)] == 1
        assert stats[(2, 3)] == 1

    def test_single_token(self):
        assert get_stats([42]) == {}

    def test_empty(self):
        assert get_stats([]) == {}

    def test_all_same(self):
        ids = [5, 5, 5, 5]
        stats = get_stats(ids)
        assert stats[(5, 5)] == 3


class TestMerge:
    def test_basic_merge(self):
        ids = [1, 2, 3, 1, 2]
        result = merge(ids, (1, 2), 99)
        assert result == [99, 3, 99]

    def test_no_match(self):
        ids = [1, 2, 3]
        result = merge(ids, (4, 5), 99)
        assert result == [1, 2, 3]

    def test_overlapping_pairs(self):
        # (1,1) appears at positions 0,1,2 but merging is non-overlapping
        ids = [1, 1, 1]
        result = merge(ids, (1, 1), 99)
        assert result == [99, 1]   # merges first pair, leftover 1 stays

    def test_empty(self):
        assert merge([], (1, 2), 99) == []


# ---------------------------------------------------------------
# Integration tests: full tokenizer
# ---------------------------------------------------------------

SAMPLE_TEXT = (
    "the cat sat on the mat. "
    "the cat is fat. "
    "a fat cat sat on a mat. " * 20   # repeat to create trainable patterns
)


@pytest.fixture(scope="module")
def trained_tokenizer():
    tok = BPETokenizer()
    tok.train(SAMPLE_TEXT, vocab_size=300, verbose=False)
    return tok


class TestBPETokenizer:
    def test_vocab_size(self, trained_tokenizer):
        # May be <= target if corpus runs out of unique pairs (early stop)
        assert 256 <= len(trained_tokenizer.vocab) <= 300

    def test_merge_count(self, trained_tokenizer):
        # Merges = actual vocab size minus the 256 base byte tokens
        assert len(trained_tokenizer.merges) == len(trained_tokenizer.vocab) - 256

    def test_encode_returns_list_of_ints(self, trained_tokenizer):
        ids = trained_tokenizer.encode("the cat")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_decode_is_inverse_of_encode(self, trained_tokenizer):
        texts = [
            "the cat sat on the mat",
            "Hello, World!",
            "tokenization",
            "1234567890",
            "the quick brown fox",
        ]
        for text in texts:
            assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text

    def test_compression_ratio_above_one(self, trained_tokenizer):
        ratio = trained_tokenizer.compression_ratio(SAMPLE_TEXT[:200])
        assert ratio > 1.0, f"Expected compression > 1x, got {ratio:.2f}x"

    def test_empty_string(self, trained_tokenizer):
        assert trained_tokenizer.encode("") == []
        assert trained_tokenizer.decode([]) == ""

    def test_unicode_text(self, trained_tokenizer):
        text = "café naïve résumé"
        decoded = trained_tokenizer.decode(trained_tokenizer.encode(text))
        assert decoded == text

    def test_base_vocab_initialized(self):
        tok = BPETokenizer()
        assert len(tok.vocab) == 256
        assert tok.vocab[65] == b"A"
        assert tok.vocab[97] == b"a"
        assert tok.vocab[0] == b"\x00"

class TestSaveLoad:
    def test_save_and_load_roundtrip(self, trained_tokenizer, tmp_path):
        """Saved and reloaded tokenizer must produce identical results."""
        path = str(tmp_path / "tokenizer.json")
        trained_tokenizer.save(path)

        tok2 = BPETokenizer()
        tok2.load(path)

        test_text = "the cat sat on a mat"
        assert trained_tokenizer.encode(test_text) == tok2.encode(test_text)
        assert trained_tokenizer.decode(
            trained_tokenizer.encode(test_text)
        ) == tok2.decode(tok2.encode(test_text))