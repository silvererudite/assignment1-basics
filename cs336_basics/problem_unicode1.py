# print(chr(0)) ## "Empty string"

# print(repr(chr(0)))

test_string = "hello! こんにちは!"
utf8_encoded = test_string.encode("utf-8")
# print(utf8_encoded)

# print(type(utf8_encoded))

# print(list(utf8_encoded))

# print(len(list(utf8_encoded)))
# print(len(test_string))

# print(utf8_encoded.decode("utf-8"))

# difference between utf-8 and utf-16
# utf16_encoded = test_string.encode("utf-16")
# print(utf16_encoded)
# print(type(utf16_encoded))

# print(list(utf16_encoded))

# print(len(list(utf16_encoded)))
# print(len(test_string))

def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

def decode_utf8_bytes_to_str_correct(bytestring: bytes):
    return "".join(list(bytestring.decode("utf-8")))

print(decode_utf8_bytes_to_str_wrong("হোসাইন".encode("utf-8")))

# two byte sequence that does not decode to any utf character
#C0 AF