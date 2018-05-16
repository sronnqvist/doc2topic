""" 
Neural topic modeling - doc2topic
Samuel Rönnqvist, TurkuNLP <saanro@utu.fi>
"""

from keras.models import Model
from keras.layers import Input, Embedding, dot, Reshape, Activation
from keras.regularizers import l1
from keras.optimizers import Adam
#from keras.preprocessing.sequence import skipgrams
import sys
import collections
import numpy as np
import matplotlib.pyplot as plt
import random
from measures import *
import csv


### Parameters
ns_rate = 2
n_topics = 25
#         0.000005
l1_doc =  0.000002
#         0.00000005
l1_word = 0.000000005
lr = 0.005
n_epochs = 10


def read_data(filename):
	""" Read data from a single file, return data and vocabulary
		Format: one document per line, tokens are space separated """
	data = []
	vocab = set()
	f = open(filename)
	while True:
		line = f.readline()
		if not line:
			break
		data.append(line.strip().lower().split())
		vocab |= set(data[-1])
	return data, vocab


def get_topic_words(top_n=10, stopwords=set()):
	stopidxs = set([token2idx[word] for word in stopwords])
	topic_words = {}
	for topic in range(wordvecs.shape[1]):
		topic_words[topic] = sorted([(x, i) for i, x in enumerate(L1normalize(wordvecs[:,topic])) if i not in stopidxs], reverse=True)[:top_n]
	return topic_words


def most_similar_words(word, n=20):
	sims = sorted([(cosine(wordvecs[token2idx[word],:], wordvecs[i,:]), i) for i in range(3, wordvecs.shape[0])], reverse=True)
	return [(idx2token[i], s) for s, i in sims[:n]]

if len(sys.argv) < 2:
	print("Usage: %s <documents file>" % sys.argv[0])
	sys.exit()


data, vocab = read_data(sys.argv[1])
token2idx = dict((t, i) for i, t in enumerate(vocab))
idx2token = dict((i, t) for i, t in enumerate(vocab))

#print("Preparing data")#, end='', flush=True)
input_docs = []
input_tokens = []
target_tokens = []
context_tokens = []
outputs = []

cntr = collections.defaultdict(lambda: 0)
for doc_id, tokens in enumerate(data):
	if doc_id % 100 == 0:
		print("\rPreparing data: %d%%" % ((doc_id+1)/len(data)*100+1), end='', flush=True)
	token_ids = [token2idx[token] for token in tokens]
	for token in tokens:
		idx = token2idx[token]
		cntr[idx] += 1
		input_tokens.append(idx)
		input_tokens += [random.randint(1, len(vocab)-1) for x in range(ns_rate)]
		input_docs += [doc_id]*(ns_rate+1)
		outputs += [1]+[0]*ns_rate
	"""
	# Alt: use keras skipgrams
	pairs, labels = skipgrams([-1]+token_ids+[-1], len(vocab)+1, window_size=1, negative_samples=ns_rate)
	input_tokens += [context_token for _, context_token in pairs if context_token != -1]
	input_docs += [doc_id for _, context_token in pairs if context_token != -1]
	outputs += [labels[i] for i in range(len(pairs)) if pairs[i][1] != -1]"""

print()

input_docs = np.array(input_docs, dtype="int32")
input_tokens = np.array(input_tokens, dtype="int32")

outputs = np.array(outputs)

emb_dim = n_topics
batch_size = 2048*4


#for nexp in range(5):
# Repeat experiment
inlayerD = Input((1,))
inlayerW = Input((1,))
EmbD = Embedding(len(data), emb_dim, input_length=1, trainable=True, activity_regularizer=l1(l1_doc), name="docvecs")
EmbW = Embedding(len(vocab), emb_dim, input_length=1, trainable=True, activity_regularizer=l1(l1_word), name="wordvecs")

embD = EmbD(inlayerD)
embDa = Activation('relu')(embD)
embD = Reshape((emb_dim, 1))(embDa)

embW = EmbW(inlayerW)
embWa = Activation('relu')(embW)
embW = Reshape((emb_dim, 1))(embWa)

#sim = dot([embD, embW], 0, normalize=True)
dot_prod = dot([embD, embW], 1, normalize=False)
dot_prod = Reshape((1,))(dot_prod)

output = Activation('sigmoid')(dot_prod)

opt = Adam(lr=lr, amsgrad=True)

model = Model(inputs=[inlayerD,inlayerW], outputs=[output])
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', fmeasure, precision])

L2 = (lambda x: np.linalg.norm(x, 2))
L1 = (lambda x: np.linalg.norm(x, 1))
L1normalize = (lambda x: x/L1(x))
cosine = (lambda a,b: np.dot(a, b)/(L2(a)*L2(b)) if sum(a) != 0 and sum(b) != 0 else 0)
relufy = np.vectorize(lambda x: max(0., x))

docvec_layerN = 2
wordvec_layerN = 3

f1s = []
log = {}
for epoch in range(0, n_epochs):
	hist = model.fit([input_docs, input_tokens], [outputs], batch_size=batch_size, verbose=1, epochs=epoch+1, initial_epoch=epoch)
	# Evaluate
	print("Sparsity")
	print("\tDoc-topic\t\tTopic-word")
	print("\tL2/L1\t>1/N\t>2/N\tL2/L1")
	print("\t%.4f\t%.4f\t%.4f\t%.4f" % (sparsity(model.layers[docvec_layerN]),
										dims_above(model.layers[docvec_layerN], 1.),
										dims_above(model.layers[docvec_layerN], 2.),
										sparsity(model.layers[wordvec_layerN])))
	# Plot first document's topic distribution
	plt.plot(relufy(L1normalize(model.layers[docvec_layerN].get_weights()[0][0])), "C0", alpha=(epoch+1)/(n_epochs+1))
	# Check convergence
	f1s += hist.history['fmeasure']
	minimum_improvement = 0.03
	if len(f1s) >= 3:
		if f1s[-1] < f1s[-2]*(1+minimum_improvement) and f1s[-1] < f1s[-3]*(1+minimum_improvement):
			print("Stopping early.")
			if f1s[-1] < 0.65:
				print("Try reduce regularization!")
			break

plt.plot(relufy(L1normalize(model.layers[docvec_layerN].get_weights()[0][0])), "C1", alpha=1)
plt.show(block=False)

log['00_DocL2L1'] = sparsity(model.layers[docvec_layerN])
log['01_DocAbove2N'] = dims_above(model.layers[docvec_layerN], 2.)

### Inspect and evaluate topic model

docvecs = model.layers[docvec_layerN].get_weights()[0]
wordvecs = model.layers[wordvec_layerN].get_weights()[0]
docvecs = relufy(docvecs)
wordvecs = relufy(wordvecs)

stopwords = set("the a of to that for and an in is from on or be by as are with may at".split())
topic_words = get_topic_words(stopwords=stopwords)

print("Topic quality")
print("\tOverlap\tPrec.\tRecall")
log['02_Overlap'] = topic_overlap(wordvecs, topic_words)
print("\t%.4f" % log['02_Overlap'], end="")
log['03_Prec'], log['04_Recall'] = topic_prec_recall(wordvecs, topic_words, cntr, stopidxs=[token2idx[w] for w in stopwords])
print("\t%.4f\t%.4f" % (log['03_Prec'], log['04_Recall']))

mean_topicdist = L1normalize(sum(docvecs)/len(docvecs))
print("Doc-topic weight mean: %.4f std %.4f" % (np.mean(mean_topicdist), np.std(mean_topicdist)))
log['05_DocStd'] = np.std(mean_topicdist)
log['06_F1'] = f1s[-1]

log['_ns_rate'] = ns_rate
log['_n_topics'] = n_topics
log['_l1_doc'] = l1_doc
log['_l1_word'] = l1_word
log['_lr'] = lr

with open("log.csv", 'a') as csvfile:
	writer = csv.DictWriter(csvfile, sorted(log.keys()))
	writer.writeheader()
	writer.writerow(log)


# Print topic words
print("\nTopic words")
for topic in topic_words:
	print("%d:" % topic, ', '.join(["%s" % idx2token[word_id] for score, word_id in topic_words[topic]]))

most_similar_words('payment')
#most_similar_words('poliisi')
