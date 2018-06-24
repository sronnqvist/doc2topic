from doc2topic import models, corpora
import sys
from keras.callbacks import LambdaCallback
import measures
import numpy as np


def custom_evaluator(model):
	""" Preview, evaluate and log during model training """
	log = {}
	## Evaluate doc-topic distribution sparsity
	doc_l2l1 = measures.sparsity(model.get_docvecs(), n=1000)
	print("Doc L2/L1: %.3f" % doc_l2l1)
	doc_peaks = measures.peak_rate(model.get_docvecs(), 2., n=1000) # Interpretable measure of sparsity: (number of dimensions > 2/n_dims)/n_dims
	print("Doc peakiness: %.3f" % doc_peaks)
	log['mDocL2L1'], log['mDocPeak'] = doc_l2l1, doc_peaks
	topic_words = model.get_topic_words()

	## Evaluate top topic words
	log['mOverlap'] = measures.topic_overlap(topic_words)
	coherences = []
	print("\nTopic words")
	for topic in topic_words:
		coherences.append(measures.pmix_coherence([word for word,_ in topic_words[topic]], model.corpus.cntr, model.corpus.cocntr, blacklist=stopwords_fi_lemma))
		print("%d (%.3f):" % (topic, coherences[-1]), ', '.join([word for word, score in topic_words[topic]]))
	log['mPMI'] = np.nanmean(coherences)
	print("Mean semantic coherence: %.3f" % log['mPMI'])
	return log


stopwords_fi_lemma = set("ja tai ei se että olla joka jos mikä mitä tämä kun eli ne hän siis jos#ei mutta kuin".split())


data = corpora.DocData(sys.argv[1], ns_rate=1, min_count=1)
#data.count_cooccs(save_to="stt_lemmas.json")
data.load_cooccs("stt_lemma_counts.json")

for f in [1, 4, 8]:
	for lr in [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]:
		model = models.Doc2Topic(data, n_topics=50, batch_size=1024*f, n_epochs=0, lr=lr, l1_doc=0.000002)
		logger = models.Logger("log_stt.csv", model, custom_evaluator)
		model.train(10, callbacks=[LambdaCallback(on_epoch_end=logger.record)])


#model.print_topic_words()
