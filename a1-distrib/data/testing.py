import nltk
import string
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')

a = "This is an amazing Movie."

lowerc = a.lower()
print(lowerc)

b = a.split()

print(a)

curr = b[0] + " "+ b[1]
print(curr)

for word in b:
    print(word , " " ,len(word))

for i in range(len(b)):
    print(i)

# cleaned = [word.translate(str.maketrans('', '', string.punctuation)) for word in b]

# print(cleaned)

cleaned = ' '.join(b)

print(b)

c = word_tokenize(cleaned)

print(c)

d = np.zeros(3, dtype=int)

print(d)

e = Counter(['simulasi', 'hi', 'nice', 'you', 'good', 'hi', 'nice'])

print(len(e))

for i in e.keys():
    print(i)
