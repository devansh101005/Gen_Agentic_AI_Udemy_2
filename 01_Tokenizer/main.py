import tiktoken

enc=tiktoken.encoding_for_model("gpt-4o")

text="Hey There! My name is Devansh Pandey"

tokens=enc.encode(text)
print("Tokens :",tokens)

# Tokens [25216, 3274, 0, 3673, 1308, 382, 11674, 616, 71, 39738, 806]

decoded =enc.decode([25216, 3274, 0, 3673, 1308, 382, 11674, 616, 71, 39738, 806])

print("Decoded :",decoded)