# doc2topic -- Neural topic modeling

This is a neural take on LDA-style topic modeling, i.e., based on a set of documents, it provides a sparse topic distribution per document. A topic is described by a distribution over words. Documents and words are points in the same latent semantic space, whose dimensions are the topics.

The implementation is based on a lightweight neural architecture and aims to be a scalable alternative to LDA. It readily makes use of GPU computation and has been tested successfully on 1M documents with 100 topics.

Getting started: `python -m tests.basic.py data/my_docs.txt`
 
## Method

The doc2topic network structure is inspired by word2vec skip-gram, where instead of modeling co-occurrences between center and context words, co-occurrences between a word and its document ID is modeled. In order to avoid heavy softmax calculation on an output layer the size of the vocabulary (or number of documents), the model is implemented as follows. 

![Architecture of doc2topic](https://github.com/sronnqvist/doc2topic/blob/master/doc2topic.svg)

The network takes as input a word ID and a document ID, which are feed through two separate embedding layers of the same dimensionality. Each embedding dimension represents a topic. The embedding layers are L1 activity regularized in order to obtain sparse representations, i.e., a parse assignment of topics. The document embeddings are more heavily regularized than the word embeddings, as sparsity is important primarily for topic-document assignments, but document and word embeddings are supposed to be comparable.

The network is trained by negative sampling, i.e., for any document both actual co-occurring words and random (supposed non-co-occurring) words are feed to the network. The two embeddings are compared by dot product, and a sigmoid activation function is applied in order to obtain values from 0 to 1. The training output label is 1 for co-occurring words and 0 for negative samples. This will push document vectors towards the vectors of the words of the document.
