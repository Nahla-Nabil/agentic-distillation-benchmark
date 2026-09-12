"""Unit tests for adbench.data.general_eval's pure functions — no network,
same testing philosophy as tests/test_prepare.py."""

from adbench.data.general_eval import filter_and_index_wikitext, sample_texts

# --- filter_and_index_wikitext ---

def test_filter_drops_blank_rows():
    raw = ["", "   ", "\n", "This is a real sentence with enough words in it to pass the filter easily."]
    kept = filter_and_index_wikitext(raw, min_word_count=5)
    assert len(kept) == 1
    assert kept[0]["text"].startswith("This is a real sentence")


def test_filter_drops_heading_only_rows():
    raw = [
        " = = Career = = \n",
        " = Robert Boulter = \n",
        "Boulter starred in two films in 2008, directed by two different filmmakers, both released to mixed reviews from critics.",
    ]
    kept = filter_and_index_wikitext(raw, min_word_count=10)
    assert len(kept) == 1
    assert "Boulter starred" in kept[0]["text"]


def test_filter_keeps_source_index_for_provenance():
    raw = ["short", "", "this row has more than enough words to clear the minimum threshold set"]
    kept = filter_and_index_wikitext(raw, min_word_count=5)
    assert kept[0]["source_index"] == 2  # its position in the ORIGINAL list


def test_filter_records_word_count():
    raw = ["one two three four five six seven eight nine ten"]
    kept = filter_and_index_wikitext(raw, min_word_count=5)
    assert kept[0]["n_words"] == 10


def test_filter_strips_whitespace():
    raw = ["   leading and trailing whitespace around this sentence right here   \n"]
    kept = filter_and_index_wikitext(raw, min_word_count=5)
    assert kept[0]["text"] == "leading and trailing whitespace around this sentence right here"


def test_filter_boundary_exactly_at_threshold_is_kept():
    raw = ["one two three four five"]  # exactly 5 words
    kept = filter_and_index_wikitext(raw, min_word_count=5)
    assert len(kept) == 1


def test_filter_empty_input():
    assert filter_and_index_wikitext([], min_word_count=10) == []


# --- sample_texts ---

def _records(n):
    return [{"text": f"text {i}", "source_index": i, "n_words": 10} for i in range(n)]


def test_sample_texts_returns_requested_count():
    sampled = sample_texts(_records(100), n_samples=10, seed=1)
    assert len(sampled) == 10


def test_sample_texts_deterministic_for_fixed_seed():
    a = sample_texts(_records(100), n_samples=10, seed=42)
    b = sample_texts(_records(100), n_samples=10, seed=42)
    assert a == b


def test_sample_texts_different_seeds_differ():
    a = sample_texts(_records(100), n_samples=10, seed=1)
    b = sample_texts(_records(100), n_samples=10, seed=2)
    assert a != b


def test_sample_texts_fewer_records_than_requested_returns_all():
    sampled = sample_texts(_records(3), n_samples=10, seed=1)
    assert len(sampled) == 3


def test_sample_texts_no_duplicates():
    sampled = sample_texts(_records(50), n_samples=20, seed=7)
    ids = [r["source_index"] for r in sampled]
    assert len(ids) == len(set(ids))
