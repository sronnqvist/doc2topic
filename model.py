from keras.models import Model
from keras.layers import Input, Embedding, dot, Reshape, Activation, Dense
from keras.regularizers import l1
from keras.optimizers import Adam
from measures import *
import numpy as np


docvec_layerN = 2
wordvec_layerN = 3
relufy = np.vectorize(lambda x: max(0., x))


def init_model(data_len, vocab_len, emb_dim, l1_doc, l1_word, lr, word_dim=None):
	inlayerD = Input((1,))
	embD = Embedding(data_len, emb_dim, input_length=1, trainable=True, activity_regularizer=l1(l1_doc), name="docvecs")(inlayerD)
	embDa = Activation('relu')(embD)
	embD = Reshape((emb_dim, 1))(embDa)

	inlayerW = Input((1,))
	if word_dim:
		embW = Embedding(vocab_len, word_dim, input_length=1, trainable=True, activity_regularizer=l1(l1_word), name="wordemb")(inlayerW)
		embWa = Dense(emb_dim, activation='relu', activity_regularizer=l1(l1_word), name="wordproj")(embW)
		embW = Reshape((emb_dim, 1))(embWa)
		#wordvec_layerN = 5
	else:
		embW = Embedding(vocab_len, emb_dim, input_length=1, trainable=True, activity_regularizer=l1(l1_word), name="wordvecs")(inlayerW)
		embWa = Activation('relu')(embW)
		embW = Reshape((emb_dim, 1))(embWa)

	#sim = dot([embD, embW], 0, normalize=True)
	dot_prod = dot([embD, embW], 1, normalize=False)
	dot_prod = Reshape((1,))(dot_prod)

	output = Activation('sigmoid')(dot_prod)

	opt = Adam(lr=lr, amsgrad=True)

	model = Model(inputs=[inlayerD,inlayerW], outputs=[output])
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', fmeasure, precision])
	return model


def get_docvecs(model, min_zero=True):
	layer_lookup = dict([(x.name,i) for i,x in enumerate(model.layers)])
	docvecs = model.layers[layer_lookup['docvecs']].get_weights()[0]
	if min_zero:
		return relufy(docvecs)
	else:
		return docvecs # Faster without relufying


def get_wordvecs(model, min_zero=True):
	layer_lookup = dict([(x.name,i) for i,x in enumerate(model.layers)])
	try:
		wordvecs = model.layers[layer_lookup['wordvecs']].get_weights()[0]
		if min_zero:
			return relufy(wordvecs)
		else:
			return wordvecs # Faster without relufying
	except KeyError:
		raise
		"""_, n_topics = model.layers[layer_lookup['docvecs']].get_weights()[0].shape
		vocab_len, _ = model.layers[layer_lookup['wordemb']].get_weights()[0].shape
		inlayerW = Input((1,))
		embW = Embedding(len(vocab), 50, input_length=1, weights=model.layers[layer_lookup['wordemb']].get_weights())(inlayerW)
		embWa = Dense(n_topics, activation='relu', weights=model.layers[layer_lookup['wordproj']].get_weights())(embW)
		wordvec_model = Model(inputs=[inlayerW], outputs=[embWa])
		return np.reshape(wordvec_model.predict(list(range(vocab_len))), (vocab_len, n_topics))"""
