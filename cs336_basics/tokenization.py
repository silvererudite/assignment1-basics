import regex as re
from collections import Counter
utf_code  = ord('牛')
# print(utf_code)
test_string = "hello! こんにちは!"
utf8_encoded = test_string.encode("utf-8")
# print(utf8_encoded)

# tokenizer vocab 
# vocab is map of tokens to integer ids, so we start with the 256 byte to int map. 
# Our final deliverable is a modified version of this vocab map that contains 
# new tokens and their int ids

# step 1 Pretokenize - computing subword level count requires going through the corpus multiple times and also redundant as
# there is possibly a lot of repititions for ex: "the" appears in text more than once we dont want to pass over the text 
# multiple time for a vocab we already seen, which is suboptimal. So we perform some engineering first. 
# Based on our context e.g language, task etc, we decide a way to soft tokenize or pretokenize to save computational time.
# initial vocab size = 256 because we have bytes from 0 - 255
# pretokenization
# input -> raw corpus output -> unique words with their frequency
# ("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
# base vocab ["b", "g", "h", "n", "p", "s", "u"]
# split all words into the symbols of base vocab, we get a dict of tuple of words and their frequency
# ("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)
# then we perform the merge operation where we count the frequency of each possible symbol pair and 
# replace those individual symbols with the newly merged pair, so vocab size increases by 1 
# and for each new merge step, the vocab size increases.
def calculate_bigram_freq(token_freq):
    bigram_freq = {}
    for token, freq in token_freq.items():
        for i in range(len(token)-1):
            current_bigram = token[i] + token[i+1]
            if current_bigram in bigram_freq:
                bigram_freq[current_bigram] += freq
            else:
                bigram_freq[current_bigram] = freq
    return bigram_freq
def get_key_with_max_value_lexicographically(d):
    """
    Returns the key with the maximum value from a dictionary.
    If multiple keys have the same maximum value, the key that comes
    first in lexicographical order is returned.

    Args:
        d (dict): The input dictionary.

    Returns:
        Any: The key with the maximum value, or None if the dictionary is empty.
    """
    if not d:
        return None

    # Sort the items based on value (descending) and then key (ascending)
    # The key for sorting is a tuple: (-value, key)
    # Sorting by -value ensures descending order for values.
    # Sorting by key ensures ascending (lexicographical) order for keys
    # when values are equal.
    sorted_items = sorted(d.items(), key=lambda item: (-item[1], item[0]))

    # The first element of the sorted list will contain the desired key-value pair
    return sorted_items[0][0]
# corpus = """low low low low low lower lower widest widest widest newest newest newest newest newest newest"""
# corpus = """low low low low low
# lower lower widest widest widest
# newest newest newest newest newest newest
# """
# part 2.4
def BPE_TRAINING(corpus):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokenized_corpus = re.findall(PAT, corpus)
    # pre_tokenized_corpus = corpus.split(" ")
    #print(pre_tokenized_corpus)
    #pretokens_corpus_with_space = [token.replace(" ", ".") for token in pre_tokenized_corpus if " " in token]

    pretokens_freq = dict(Counter(pre_tokenized_corpus))
    # initial vocabulary
    vocab = {bytes([i]): i for i in range(256)} 
    vocab['<|endoftext|>'] = 256
    print("length of initial vocab ", len(vocab))

    num_merges = 6

    for merge in range(num_merges):
        # print(vocab.keys())
        print(f"=============== step {merge + 1} =============")
        pretokens_freq = {tuple(token): freq for token, freq in pretokens_freq.items()}
        print(f"pretokens in step {merge + 1} ", pretokens_freq)

        # pretokens_chars = [tuple(key) for key in pretokens_freq]
        # print(pretokens_chars)
        # this dict data structure is crucial to avoid multiple passes over the corpus, as we can now find 
        # the pair/ bigram freq using the characters from the keys of the dict 
        
        bigram_freq = calculate_bigram_freq(pretokens_freq)
        print(f"pretokens_bigram_freq in {merge + 1} step: ", bigram_freq)

        # def get_max_pair(pretokens_bigram_freq):
        #     return max(pretokens_bigram_freq, key=pretokens_bigram_freq.get)
        

        # print("max pair in the order inserted: ", get_max_pair(bigram_freq))
        new_token = get_key_with_max_value_lexicographically(bigram_freq)
        print(f"max pair lexicographically sorted in {merge + 1} step : ", new_token)

        # def merge_step(vocab, new_token, pretokens_freq):
        #     vocab[new_token.encode("utf-8")] = len(vocab) + 1
        #     for word, freq in pretokens_freq.items():
        #         for i in range(len(word)-1):
        #             if len(word) > len(new_token) and (word[i] + word[i+1] == new_token):
        #                 new_tuple =  word[:i] + (new_token,) + word[i+len(new_token):]
        #                 pretokens_freq.pop(word)
        #                 pretokens_freq[new_tuple] = freq
        #     return pretokens_freq
        def merge_step(vocab, new_token, pretokens_freq):
            next_id = len(vocab)
            if new_token not in vocab:
                vocab[new_token] = next_id
                next_id += 1
            new_pretokens_freq = {}
            
            for word, freq in pretokens_freq.items():
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word)-1 and (word[i] + word[i+1]) == new_token:
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_pretokens_freq[tuple(new_word)] = freq

            return new_pretokens_freq

        
        # add the new merged token to the pretokens container
        pretokens_freq = merge_step(vocab, new_token, pretokens_freq)
        print(f"length of vocab after merging in {merge+1} step ", len(vocab))
        print(f"pretokens frequency after {merge + 1} step ", pretokens_freq)
        print("=====================================")
        if merge == 5:
            for token, idx in vocab.items():
                try:
                    print(idx, token)
                except UnicodeDecodeError:
                    print(idx, repr(token))  # fallback for non-printable bytes
        return vocab


