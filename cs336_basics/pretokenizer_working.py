import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

text = "some text that i'll pre-tokenize"
tokens_list = [token.group() for token in re.finditer(PAT, text)]
print(tokens_list)

# for token in re.finditer(PAT, text):
#     print(dir(token))
#     break

# for token in re.finditer(PAT, text):
#     help(token.start())
#     break

for token in re.finditer(PAT, text):
    print(f"Word: {token.group():<10} Starts at: {token.start()} | Ends at: {token.end()}")