from keras.models import Model
from keras.layers import Input, Embedding, dot, Reshape, Activation, Dense
from keras.regularizers import l1
from keras.optimizers import Adam
from sklearn.metrics.pairwise import cosine_similarity
from .measures import fmeasure

from os.path import isfile
import numpy as np
import json, csv
import heapq


# Config GPU memory usage
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session, clear_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = -1
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


L2 = (lambda x: np.linalg.norm(x, 2))
L1 = (lambda x: np.linalg.norm(x, 1))
L1normalize = (lambda x: x/L1(x))
cosine = (lambda a,b: np.dot(a, b)/(L2(a)*L2(b)) if sum(a) != 0 and sum(b) != 0 else 0)
relufy = np.vectorize(lambda x: max(0., x))


class Doc2Topic:
	""" doc2topic model class """
	def __init__(self, corpus, n_topics=20, batch_size=1024*6, n_epochs=5, lr=0.015, l1_doc=0.000002, l1_word=0.000000015, word_dim=None):
		self.corpus = corpus
		self.params = {	'Ntopics':	n_topics,
						'Ndocs':	self.corpus.n_docs,
						'BS': 		batch_size,
						'LR': 		lr,
						'L1doc':	l1_doc,
						'L1word':	l1_word,
						'NS':		self.corpus.ns_rate}
		self.topic_words = None
		self.wordvecs = None
		self.docvecs = None

		inlayerD = Input((1,))
		embD = Embedding(self.corpus.n_docs, n_topics, input_length=1, trainable=True, activity_regularizer=l1(l1_doc), name="docvecs")(inlayerD)
		embDa = Activation('relu')(embD)
		embD = Reshape((n_topics, 1))(embDa)

		inlayerW = Input((1,))
		if word_dim: # Experimental setting: extra dense layer for projecting word vectors onto document vector space
			embW = Embedding(self.corpus.vocab_size, word_dim, input_length=1, trainable=True, activity_regularizer=l1(l1_word), name="wordemb")(inlayerW)
			embWa = Dense(emb_dim, activation='relu', activity_regularizer=l1(l1_word), name="wordproj")(embW)
			embW = Reshape((n_topics, 1))(embWa)
		else:
			embW = Embedding(self.corpus.vocab_size, n_topics, input_length=1, trainable=True, activity_regularizer=l1(l1_word), name="wordvecs")(inlayerW)
			embWa = Activation('relu')(embW)
			embW = Reshape((n_topics, 1))(embWa)

		#sim = dot([embD, embW], 0, normalize=True)
		dot_prod = dot([embD, embW], 1, normalize=False)
		dot_prod = Reshape((1,))(dot_prod)

		output = Activation('sigmoid')(dot_prod)

		opt = Adam(lr=lr, amsgrad=True)

		self.model = Model(inputs=[inlayerD,inlayerW], outputs=[output])
		self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[fmeasure])
		self.layer_lookup = dict([(x.name,i) for i,x in enumerate(self.model.layers)])

		self.train(n_epochs=n_epochs)


	def train(self, n_epochs, callbacks=[]):
		self.docvecs = None
		self.wordvecs = None
		self.history = self.model.fit([self.corpus.input_docs, self.corpus.input_tokens], [self.corpus.outputs], batch_size=self.params['BS'], verbose=1, epochs=n_epochs, callbacks=callbacks)


	def save(self, filename):
		json.dump(self.corpus.idx2token, open("%s.vocab" % filename,'w')) # Save token index mapping
		self.model.save(filename)


	def get_docvecs(self, min_zero=True):
		if self.docvecs is None:
			self.docvecs = self.model.layers[self.layer_lookup['docvecs']].get_weights()[0]
			if min_zero: # Faster without relufying
				self.docvecs = relufy(self.docvecs)
		return self.docvecs


	def get_wordvecs(self, min_zero=True):
		if self.wordvecs is None:
			self.wordvecs = self.model.layers[self.layer_lookup['wordvecs']].get_weights()[0]
			if min_zero:
				self.wordvecs = relufy(self.wordvecs)
		return self.wordvecs
		# For dense projection layer (obsolete)
		"""_, n_topics = model.layers[layer_lookup['docvecs']].get_weights()[0].shape
		vocab_len, _ = model.layers[layer_lookup['wordemb']].get_weights()[0].shape
		inlayerW = Input((1,))
		embW = Embedding(len(vocab), 50, input_length=1, weights=model.layers[layer_lookup['wordemb']].get_weights())(inlayerW)
		embWa = Dense(n_topics, activation='relu', weights=model.layers[layer_lookup['wordproj']].get_weights())(embW)
		wordvec_model = Model(inputs=[inlayerW], outputs=[embWa])
		return np.reshape(wordvec_model.predict(list(range(vocab_len))), (vocab_len, n_topics))"""


	def get_topic_words(self, top_n=10, stopwords=set()):
		self.get_wordvecs()
		topic_words = {}
		for topic in range(self.wordvecs.shape[1]):
			topic_words[topic] = heapq.nlargest(top_n+len(stopwords), enumerate(L1normalize(self.wordvecs[:,topic])), key=lambda x:x[1])
			topic_words[topic] = [(self.corpus.idx2token[idx], score) for idx, score in topic_words[topic] if self.corpus.idx2token[idx] not in stopwords]
		self.topic_words = topic_words
		return topic_words


	def print_topic_words(self, top_n=10, stopwords=set()):
		if self.topic_words is None:
			self.get_topic_words(top_n=top_n, stopwords=stopwords)
		print("Topic words")
		for topic in self.topic_words:
			print("%d:" % topic, ', '.join(["%s" % word for word, score in self.topic_words[topic]]))


	def most_similar_words(self, word, n=20):
		self.get_wordvecs()
		idx = self.corpus.token2idx[word]
		sims = heapq.nlargest(n, enumerate(cosine_similarity(self.wordvecs[idx:idx+1,:], self.wordvecs)[0]), key=lambda x:x[1])
		return [(self.corpus.idx2token[i], s) for i, s in sims]


class Logger:
	def __init__(self, filename, model, evaluator):
		self.filename = filename
		self.evaluator = evaluator
		self.model = model
		self.log = dict([('p%s'%p, v) for p, v in model.params.items()])

	def record(self, epoch, logs):
		self.log['_Epoch'] = epoch
		self.log['_Loss'] = logs['loss']
		self.log['_F1'] = logs['fmeasure']
		self.log.update(self.evaluator(self.model))
		self.write()

	def write(self):
		file_exists = isfile(self.filename)
		with open(self.filename, 'a') as csvfile:
			writer = csv.DictWriter(csvfile, sorted(self.log.keys()))
			if not file_exists:
				writer.writeheader()
			writer.writerow(self.log)
