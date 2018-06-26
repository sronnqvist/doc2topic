import collections
import numpy as np
import random
import json


class DocData:
	def __init__(self, filename, min_count=5, ns_rate=2):
		self.docs, self.cntr = self.read_docs_file(filename)
		self.n_docs = len(self.docs)
		self.vocab_size = np.nan
		self.ns_rate = ns_rate
		self.prepare(min_count=min_count, ns_rate=ns_rate)


	def read_docs_file(self, filename, lowercase=True):
		""" Read document data from a single file, return data and word counter
			Input format: one document per line, tokens space separated """
		data = []
		cntr = collections.defaultdict(lambda: 0)
		print("Reading documents...", end='', flush=True)
		f = open(filename)

		while True:
			line = f.readline()
			if not line:
				break
			if lowercase:
				line = line.lower()
			data.append(line.strip().split())
			for token in data[-1]:
				cntr[token] += 1
			if len(data) % 100 == 0:
				print("\rReading documents: %d" % len(data), end='', flush=True)

		print()
		return data, cntr


	def prepare(self, min_count=5, ns_rate=2, replace=False):
		""" Prepare training data and vocabulary mappings from documents """
		self.vocab_size = len([cnt for cnt in self.cntr.values() if cnt > min_count])
		print("Vocabulary size: %d" % self.vocab_size)

		self.input_docs, self.input_tokens, self.outputs = [], [], []
		self.token2idx = collections.defaultdict(lambda: len(self.token2idx))
		for doc_id, tokens in enumerate(self.docs):
			if doc_id % 100 == 0:
				print("\rPreparing data: %d%%" % ((doc_id+1)/len(self.docs)*100+1), end='', flush=True)
			# Filter tokens by frequency and map them to IDs (creates mapping table on the fly)
			token_ids = [self.token2idx[token] for token in tokens if self.cntr[token] > min_count]
			for i, idx in enumerate(token_ids):
				self.input_tokens.append(idx)
				self.input_tokens += [random.randint(1, self.vocab_size-1) for x in range(ns_rate)]
				self.input_docs += [doc_id]*(ns_rate+1)
				self.outputs += [1]+[0]*ns_rate

		print()
		self.input_docs = np.array(self.input_docs, dtype="int32")
		self.input_tokens = np.array(self.input_tokens, dtype="int32")
		self.outputs = np.array(self.outputs)

		self.idx2token = dict([(i,t) for t,i in self.token2idx.items()])
		if replace:
			del self.docs


	def count_cooccs(self, save_to=None):
		""" Count word co-occurrences for PMI coherence evaluation """
		self.cocntr = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
		print("Counting word co-occurrences...")
		for tokens in self.docs:
			for i, token1 in enumerate(tokens[:-1]):
				for token2 in tokens[i+1:min(i+110,len(tokens))]:
					t1, t2 = sorted([token1, token2])
					self.cocntr[t1][t2] += 1

		if save_to:
			json.dump([self.cntr, self.cocntr], open(save_to, 'w'))


	def load_cooccs(self, filename):
		""" Load word co-occurrence counts for PMI coherence evaluation """
		print("Loading word co-occurrence data...")
		self.cntr, self.cocntr = json.load(open(filename))
