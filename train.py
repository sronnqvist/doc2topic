"""
Neural topic modeling - doc2topic
Samuel Rönnqvist, TurkuNLP <saanro@utu.fi>
"""

# Config GPU memory usage
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session, clear_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = -1
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

#from keras.preprocessing.sequence import skipgrams
from model import init_model
from measures import *
import sys
import collections
import numpy as np
import matplotlib.pyplot as plt
import random
import csv
from os.path import isfile


### Hyperparameters
n_topics   = 100 # Increase with number of documents
l1_doc     = 0.000002#15
l1_word    = 0.000000015
lr         = 0.015
batch_size = 10*1024 #~6-8/10k for 10-100k documents, too high might slow down learning
ns_rate    = 2 # Negative sampling rate, 1-2 recommended
n_epochs   = 50 # Max
min_count  = 5 # Minimum word count


def read_data(filename):
	""" Read data from a single file, return data and vocabulary
		Format: one document per line, tokens are space separated """
	data = []
	vocab = set()
	cntr = collections.defaultdict(lambda: 0)
	print("Reading documents...", end='', flush=True)
	f = open(filename)

	while True:
		line = f.readline()
		if not line:
			break
		data.append(line.strip().lower().split())
		#vocab |= set(data[-1])
		for token in data[-1]:
			cntr[token] += 1
		if len(data) % 100 == 0:
			print("\rReading documents: %d" % len(data), end='', flush=True)

	print()
	return data, cntr#vocab


def get_topic_words(top_n=10, stopwords=set()):
	stopidxs = set([token2idx[word] for word in stopwords])
	topic_words = {}
	for topic in range(wordvecs.shape[1]):
		topic_words[topic] = sorted([(x, i) for i, x in enumerate(L1normalize(wordvecs[:,topic])) if i not in stopidxs], reverse=True)[:top_n]
	return topic_words


def most_similar_words(word, n=20):
	sims = sorted([(cosine(wordvecs[token2idx[word],:], wordvecs[i,:]), i) for i in range(3, wordvecs.shape[0])], reverse=True)
	return [(idx2token[i], s) for s, i in sims[:n]]


def write_log(log, filename="log.csv"):
	file_exists = isfile(filename)
	with open(filename, 'a') as csvfile:
		writer = csv.DictWriter(csvfile, sorted(log.keys()))
		if not file_exists:
			writer.writeheader()
		writer.writerow(log)


L2 = (lambda x: np.linalg.norm(x, 2))
L1 = (lambda x: np.linalg.norm(x, 1))
L1normalize = (lambda x: x/L1(x))
cosine = (lambda a,b: np.dot(a, b)/(L2(a)*L2(b)) if sum(a) != 0 and sum(b) != 0 else 0)
relufy = np.vectorize(lambda x: max(0., x))


#### Main begin

if len(sys.argv) < 2:
	print("Usage: %s <documents file>" % sys.argv[0])
	sys.exit()


### Prepare data
#print("Reading document data...")
data, cntr = read_data(sys.argv[1])
vocab_len = len([cnt for cnt in cntr.values() if cnt > min_count])
print("Vocabulary size: %d" % vocab_len)

#cntr = collections.defaultdict(lambda: 0)
#print("Counting words...")
#cntr, cocntr = count_words(data, save_to="stt_lemmas.json")

#print("Loading word count data...")
#cntr, cocntr = load_counts("stt_lemma_counts_100k.json")

input_docs, input_tokens, outputs = [], [], []
token2idx = collections.defaultdict(lambda: len(token2idx))
for doc_id, tokens in enumerate(data):
	if doc_id % 100 == 0:
		print("\rPreparing data: %d%%" % ((doc_id+1)/len(data)*100+1), end='', flush=True)
	# Filter tokens by frequency and map them to IDs (creates mapping table on the fly)
	token_ids = [token2idx[token] for token in tokens if cntr[token] > min_count]
	for i, idx in enumerate(token_ids):
		input_tokens.append(idx)
		input_tokens += [random.randint(1, vocab_len-1) for x in range(ns_rate)]
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

idx2token = dict([(i,t) for t,i in token2idx.items()])

### Modeling
#docvec_layerN = 2
#wordvec_layerN = 3

log = {}

stopwords_fi_lemma = set("ja tai ei se että olla joka jos mikä mitä tämä kun eli ne hän siis jos#ei mutta kuin".split())

# Create model with given settings
model = init_model(len(data), vocab_len, n_topics, l1_doc, l1_word, lr)

log['p_Ndocs'] = len(data)
log['p_BS'] = batch_size
log['p_NSrate'] = ns_rate
log['p_Ntopics'] = n_topics
log['p_L1doc'] = l1_doc
log['p_L1word'] = l1_word
log['p_LR'] = lr
# Print parameter names and values
print('\t'.join([name for name in log if name[0] == 'p']))
print('\t'.join([str(log[name]) for name in log if name[0] == 'p']))

f1s = []
pmis = []
for epoch in range(0, n_epochs):
	#hist = model.fit([input_docs[:hyper_batch_size], input_tokens[:hyper_batch_size]], [outputs[:hyper_batch_size]], batch_size=batch_size, verbose=1, epochs=epoch+1, initial_epoch=epoch)#, validation_split=0.05)
	hist = model.fit([input_docs, input_tokens], [outputs], batch_size=batch_size, verbose=1, epochs=epoch+1, initial_epoch=epoch)#, validation_split=0.05)
	#input_docs = input_docs[hyper_batch_size:]+input_docs[:hyper_batch_size]
	#input_tokens = input_tokens[hyper_batch_size:]+input_tokens[:hyper_batch_size]
	#outputs = outputs[hyper_batch_size:]+outputs[:hyper_batch_size]
	if epoch % 5 != 4:
		continue
	"""docvecs = model.layers[docvec_layerN].get_weights()[0]
	wordvecs = model.layers[wordvec_layerN].get_weights()[0]
	docvecs = relufy(docvecs)
	wordvecs = relufy(wordvecs)"""
	docvecs = get_docvecs(model)
	wordvecs = get_wordvecs(model)

	# Evaluate
	print("Sparsity")
	print("\tDoc-topic\tTopic-word")
	print("\tL2/L1\t>2/N\tL2/L1") # Todo: Normalized Above 2/N measure: |{x|x>2/N}|/N
	doc_sparsity = sparsity(model.layers[docvec_layerN], n=1000)
	word_sparsity = np.nan#sparsity(model.layers[wordvec_layerN], n=1000)
	above2N = dims_above(model.layers[docvec_layerN], 2.) # Interpretable measure of sparsity: number of dimensions > 2/n_dims
	print("\t%.4f\t%.4f\t%.4f" % (doc_sparsity, above2N, word_sparsity))
	log['a_Epoch'], log['m_DocL2L1'], log['m_DocAbove2N'] = epoch, doc_sparsity, above2N
	topic_words = get_topic_words()
	log['m_tOverlap'] = topic_overlap(wordvecs, topic_words)
	log['m_tPrec'],	log['m_tRecall'] = topic_prec_recall(topic_words, idx2token, cntr, stopidxs=set(), n_freq_words=n_topics*10)#[token2idx[w] for w in stopwords])
	log['m_tWordy'], log['m_tStopy'] = topic_wordiness(topic_words, idx2token), topic_stopwordiness(topic_words, idx2token, stopwords_fi_lemma)
	log['z_F1'], log['z_Acc'] = hist.history['fmeasure'][0], hist.history['acc'][0]

	coherences = []
	print("\nTopic words")
	for topic in topic_words:
		coherences.append(pmix_coherence([idx2token[i] for _, i in topic_words[topic]], cntr, cocntr, blacklist=stopwords_fi_lemma))
		print("%d (%.3f):" % (topic, coherences[-1]), ', '.join(["%s" % idx2token[word_id] for score, word_id in topic_words[topic]]))

	log['m_PMI'] = np.nanmean(coherences)
	print("Mean semantic coherence: %.3f" % log['m_PMI'])
	pmis.append(log['m_PMI'])
	write_log(log, "log_stt_coh4.csv")
	# Plot first document's topic distribution
	plt.plot(relufy(L1normalize(model.layers[docvec_layerN].get_weights()[0][0])), "C0", alpha=(epoch+1)/(n_epochs+1))
	plt.show(block=False)
	# Check convergence
	f1s += hist.history['fmeasure']


# Print topic words
print("Topic overlap:", log['m_tOverlap']) # Topic-topic overlap in top-10 words; good range: 0-0.15
print("Topic precision:", log['m_tPrec'])
print("Topic recall:",	log['m_tRecall']) # Topic recall: how well top-10 topic words cover top-10*n_topics most frequent words; good range: 0.3-1
print("Topic wordiness:",	log['m_tWordy']) # Rate of alpha tokens (i.e., good topic words compared to numbers and punctuations); good range: 0.94-1
print("Topic stop wordiness:",	log['m_tStopy']) # Rate of stop words; good range: 0-0.05


### Inspect and evaluate topic model (obsolete stuff)
"""
stopwords = set("the a of to that for and an in is from on or be by as are with may at".split())
stopwords = set()
topic_words = get_topic_words(stopwords=stopwords)
"""
"""
print("Topic quality")
print("\tOverlap\tPrec.\tRecall")
log['02_Overlap'] = topic_overlap(wordvecs, topic_words)
print("\t%.4f" % log['02_Overlap'], end="")
log['03_Prec'], log['04_Recall'] = topic_prec_recall(wordvecs, topic_words, cntr, stopidxs=[token2idx[w] for w in stopwords])
print("\t%.4f\t%.4f" % (log['03_Prec'], log['04_Recall']))
"""
#mean_topicdist = L1normalize(sum(docvecs)/len(docvecs))
#print("Doc-topic weight mean: %.4f std %.4f" % (np.mean(mean_topicdist), np.std(mean_topicdist)))
#log['05_DocStd'] = np.std(mean_topicdist)
#log['06_F1'] = f1s[-1]


#log['07_coherence'] = topic_coherence(topic_words, idx2token)

#most_similar_words('payment')
#most_similar_words('poliisi')
