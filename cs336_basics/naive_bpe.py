from collections import Counter
import pprint

text = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest"""

# initialize the vocabulary
# 1. Initialize the Base Vocabulary with the 256 raw bytes
# (We represent each raw byte value 0-255 as a single-byte object)
vocabulary = ["<|endoftext|>"] + [bytes([i]) for i in range(256)]

pretokenized_text = text.split()
bytes_of_pretokenized_text = []
for word in pretokenized_text:
    exploded_bytes = tuple(bytes([b]) for b in word.encode("utf-8"))
    bytes_of_pretokenized_text.append(exploded_bytes)
# build a frequency dictionary
word_freq = Counter(bytes_of_pretokenized_text)

pprint.pprint(dict(word_freq))
def merge_vocab(pair, vocab_in):
    vocab_out = {}
    p0, p1 = pair
    for word, freq in vocab_in.items():
        new_word = []   
        i = 0

        while i < len(word):
            if i < len(word)-1 and word[i] == p0 and word[i+1] == p1:
                new_word.append(p0 + p1)
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        vocab_out[tuple(new_word)] = freq

    return vocab_out

merges_tracked = []
num_merges = 6

for _ in range(num_merges):
    # take successive pairs and count their frequencies
    merge_dict = {}
    for word, freq in word_freq.items():
        for i in range(len(word)-1):
            probable_pair = (word[i], word[i+1])
            merge_dict[probable_pair] = merge_dict.get(probable_pair, 0) + freq
    if not merge_dict:
        break
    # find pairs with highest frequency and solve tie by selecting lexicoraphically greater pair
    max_freq = max(merge_dict.values())
    highest_freq_pairs = sorted([k for k,v in merge_dict.items() if v == max_freq], key=str, reverse=True)
    best_pair = highest_freq_pairs[0]
    new_token = best_pair[0] + best_pair[1]

    pair_str = f"{best_pair[0].decode()} {best_pair[1].decode()}"
    merges_tracked.append(pair_str)

    vocabulary.append(new_token)
    word_freq = merge_vocab(best_pair, word_freq)

print("--- Merges Performed ---")
print(merges_tracked)

print("\n--- Final 8 New Vocabulary Additions ---")
# The last 6 elements are our merges, showing the evolution
pprint.pprint([tok.decode('utf-8') if isinstance(tok, bytes) else tok for tok in vocabulary[-6:]])

