from cs336_basics.train_bpe import BPE_TRAINER
import json
from pathlib import Path


def serialize_bytes_token(token: bytes) -> str:
    """
    Convert bytes to a readable string for JSON inspection.
    Uses latin-1 so every byte 0-255 maps safely to one character.
    """
    return token.decode("latin-1")


def save_bpe_artifacts(vocab, merges, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # vocab: dict[int, bytes] -> dict[str, str]
    serializable_vocab = {
        str(token_id): serialize_bytes_token(token_bytes)
        for token_id, token_bytes in vocab.items()
    }

    # merges: list[tuple[bytes, bytes]] -> list[list[str, str]]
    serializable_merges = [
        [
            serialize_bytes_token(left),
            serialize_bytes_token(right),
        ]
        for left, right in merges
    ]

    with open(output_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(serializable_vocab, f, ensure_ascii=False, indent=2)

    with open(output_dir / "merges.json", "w", encoding="utf-8") as f:
        json.dump(serializable_merges, f, ensure_ascii=False, indent=2)

    print(f"Saved vocab to {output_dir / 'vocab.json'}")
    print(f"Saved merges to {output_dir / 'merges.json'}")

def train_bpe_on_tinystories(input_path, vocab_size, special_tokens):
    trainer = BPE_TRAINER(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
        )
    return trainer.train_bpe()

import time
import tracemalloc


def inspect_bpe_training(input_path, vocab_size, special_tokens):
    tracemalloc.start()
    start_time = time.perf_counter()

    vocab, merges = train_bpe_on_tinystories(
        input_path,
        vocab_size,
        special_tokens,
    )

    end_time = time.perf_counter()
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed_time = end_time - start_time

    longest_token_id, longest_token = max(
        vocab.items(),
        key=lambda item: len(item[1])
    )

    print("=== BPE Training Report ===")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print(f"Current memory: {current_memory / 1024 / 1024:.2f} MB")
    print(f"Peak memory: {peak_memory / 1024 / 1024:.2f} MB")
    print(f"Final vocab size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")

    print("\n=== Longest Token ===")
    print(f"Token ID: {longest_token_id}")
    print(f"Length in bytes: {len(longest_token)}")
    print(f"Raw bytes: {longest_token}")

    try:
        print(f"Decoded token: {longest_token.decode('utf-8')}")
    except UnicodeDecodeError:
        print("Decoded token: <not valid UTF-8 by itself>")

    return vocab, merges

vocab, merges = inspect_bpe_training(
    "../data/TinyStoriesV2-GPT4-valid.txt",
    10000,
    ["<|endoftext|>"],
)
save_bpe_artifacts(vocab, merges, output_dir="./tiny_stories_valid_bpe_debug")