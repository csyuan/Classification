from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import numpy as np
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# model = Word2Vec.load("./word2vec_gensim")

#
# X = model[model.wv.vocab]

word_set = set()
f_word = open("./words/movie_comments_t_keyword.csv", "r", encoding="utf-8")
for frline in f_word:
    line = frline.strip()
    linearr = line.split("\t")
    word = linearr[0]
    count = int(linearr[1])
    if count > 600:
        word_set.add(word)

print(len(word_set))
print(word_set)

fr = open("./word2vec_org", "r",encoding="utf-8")
labels = []
tokens = []
count = 0
for frline in fr:
    line = frline.strip()
    lineArr = line.split(" ")
    word = lineArr[0]
    vec = lineArr[1:]
    if word in word_set:
        s_vec = np.array(vec)
        labels.append(word)
        tokens.append(s_vec)

print(len(labels))
print(len(tokens))
tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2000, random_state=25)
new_values = tsne_model.fit_transform( tokens)

x = []
y = []
for value in new_values:
    x.append(value[0])
    y.append(value[1])

plt.figure(figsize=(10, 10))
for i in range(len(x)):
    plt.scatter(x[i],y[i])
    plt.annotate(labels[i],
                 xy=(x[i], y[i]),
                 xytext=(2, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
plt.savefig("./figure_movie")
plt.show()
