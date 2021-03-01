import re
import numpy as np
from scipy.spatial import distance

with open('cat.txt', 'r') as f:
    arr = f.read().lower().splitlines()

texts = []

for text in arr:
    t1 = re.split('[^a-z]', text)
    texts.append([x for x in t1 if x])

for row in texts:
    print(row)

allTexts = []
for arr in texts:
    allTexts = allTexts + arr

words = {}
numb = 0
for w in allTexts:
    if w not in words:
        words[w] = numb
        numb += 1


print("Слова")
print(words)
print("-------------------------------")

matrix = np.zeros((len(texts), len(words)))

for word in words:
    m = 0
    for text in texts:
        matrix[m][words[word]] = text.count(word)
        m += 1

for row in matrix:
    print(row)

res = {}

i = 0
for row in matrix:
    res[i] = distance.cosine(matrix[0], row)
    i += 1
print(res)

print("След матрицы: ")
print(np.trace(matrix))

del res[0]
mem = min(res.items(), key=lambda x: x[1])
del res[mem[0]]
kek = min(res.items(), key=lambda x: x[1])

print("Самое близкое предложение: ")
print(mem)
print("Второе близкое предложение: ")
print(kek)
