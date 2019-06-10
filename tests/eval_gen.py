from doc2topic import models, corpora, measures
import sys
from keras.callbacks import LambdaCallback
import numpy as np


def custom_evaluator(model):
	""" Preview, evaluate and log during model training """
	log = {}
	"""
	## Evaluate doc-topic distribution sparsity
	doc_l2l1 = measures.sparsity(model.get_docvecs(), n=1000)
	print("Doc L2/L1: %.3f" % doc_l2l1)
	doc_peaks = measures.peak_rate(model.get_docvecs(), 2., n=1000) # Interpretable measure of sparsity: (number of dimensions > 2/n_dims)/n_dims
	print("Doc peakiness: %.3f" % doc_peaks)
	log['mDocL2L1'], log['mDocPeak'] = doc_l2l1, doc_peaks
	"""
	topic_words = model.get_topic_words()

	## Evaluate top topic words
	log['mOverlap'] = measures.topic_overlap(topic_words)
	coherences = []
	print("\nTopic words")
	for topic in list(topic_words.keys())[:15]:
		coherences.append(measures.pmix_coherence([word for word,_ in topic_words[topic]], model.corpus.cntr, model.corpus.cocntr, blacklist=stopwords_fi_lemma))
		print("%d (%.3f):" % (topic, coherences[-1]), ', '.join([word for word, score in topic_words[topic]]))
	log['mPMI'] = np.nanmean(coherences)
	print("Mean semantic coherence: %.3f" % log['mPMI'])
	model.save("all_stt_topics.model")
	return log


stopwords_fi_lemma = set("ja tai ei se että olla joka jos mikä mitä tämä kun eli ne hän siis jos#ei mutta kuin".split())


data = corpora.DocData(sys.argv[1], ns_rate=2, min_count=10, with_generator=True)
#data0 = corpora.DocData(sys.argv[1], ns_rate=2, min_count=10, with_generator=False)

#data.count_cooccs(save_to="stt_lemmas.json")
data.load_cooccs("stt_lemma_counts.json")

f=10
lr=0.015
feeder = models.DataGenerator(data, batch_size=1024*f, n_passes=4, ns_rate=2)
model = models.Doc2Topic(data, n_topics=200, batch_size=1024*f, n_epochs=0, lr=lr, l1_doc=0.0000002, l1_word=0.000000015, generator=feeder)
logger = models.Logger("log_stt_full.csv", model, custom_evaluator)
model.train(4, callbacks=[LambdaCallback(on_epoch_end=logger.record)])


#model.print_topic_words()
