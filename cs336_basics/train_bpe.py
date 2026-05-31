from collections import Counter
import regex as re

class BPE_TRAINER:
    def __init__(self, input_path: str, vocab_size: int, special_tokens: list[str]):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def _initialize_vocab(self) -> dict[int, bytes]:
        """
        Base vocabulary:
        IDs 0-255 are raw bytes.
        Special tokens are added after that.
        """
        vocab = {i: bytes([i]) for i in range(256)}

        next_id = 256
        for token in self.special_tokens:
            vocab[next_id] = token.encode("utf-8")
            next_id += 1

        return vocab
    
    def _split_on_special_tokens(self, text: str) -> list[str]:
        """
        Split text by special tokens.

        The special tokens themselves are removed from training chunks,
        so they do not contribute to merge statistics.
        """
        if not self.special_tokens:
            return [text]

        # Sort by length descending to avoid partial matching problems
        # Example: if "<|end|>" and "<|endoftext|>" both exist.
        escaped_tokens = [
            re.escape(tok)
            for tok in sorted(self.special_tokens, key=len, reverse=True)
        ]

        pattern = "(" + "|".join(escaped_tokens) + ")"

        pieces = re.split(pattern, text)

        # Remove special tokens from training data
        chunks = [
            piece
            for piece in pieces
            if piece and piece not in self.special_tokens
        ]

        return chunks
    def _pretokenize(self, text: str) -> Counter[tuple[bytes, ...]]:
        """
        Convert text into frequency dictionary of byte-token tuples.

        Example:
            "low low" ->
            {
                (b'l', b'o', b'w'): 2
            }
        """
        word_freq = Counter()

        chunks = self._split_on_special_tokens(text)

        for chunk in chunks:
            for match in re.finditer(self.PAT, chunk):
                token = match.group(0)
                byte_tuple = tuple(bytes([b]) for b in token.encode("utf-8"))
                word_freq[byte_tuple] += 1

        return word_freq
    
    def _get_pair_counts(
        self,
        word_freq: Counter[tuple[bytes, ...]]
    ) -> Counter[tuple[bytes, bytes]]:
        """
        Count adjacent byte-token pairs.
        """
        pair_counts = Counter()

        for word, freq in word_freq.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += freq

        return pair_counts
    
    def _merge_vocab(
        self,
        pair: tuple[bytes, bytes],
        word_freq: Counter[tuple[bytes, ...]]
    ) -> Counter[tuple[bytes, ...]]:
        """
        Replace every occurrence of pair with the merged token.
        """
        new_word_freq = Counter()
        p0, p1 = pair

        for word, freq in word_freq.items():
            new_word = []
            i = 0

            while i < len(word):
                if (
                    i < len(word) - 1
                    and word[i] == p0
                    and word[i + 1] == p1
                ):
                    new_word.append(p0 + p1)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            new_word_freq[tuple(new_word)] += freq

        return new_word_freq

    def train_bpe(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        Train byte-level BPE.

        Returns:
            vocab: dict[int, bytes]
            merges: list[tuple[bytes, bytes]]
        """
        vocab = self._initialize_vocab()
        merges = []

        with open(self.input_path, "r", encoding="utf-8") as f:
            text = f.read()

        word_freq = self._pretokenize(text)

        next_token_id = len(vocab)

        while len(vocab) < self.vocab_size:
            pair_counts = self._get_pair_counts(word_freq)

            if not pair_counts:
                break

            # Pick pair with highest frequency.
            # Tie-break: lexicographically greater pair.
            best_pair = max(
                pair_counts,
                key=lambda pair: (pair_counts[pair], pair)
            )

            new_token = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            vocab[next_token_id] = new_token
            next_token_id += 1

            word_freq = self._merge_vocab(best_pair, word_freq)

        return vocab, merges
