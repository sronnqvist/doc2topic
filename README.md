# doc2topic -- Neural topic modeling

This is a neural take on LDA-style topic modeling, i.e., based on a set of documents, it provides a sparse topic distribution for each document. Words are likewise assigned to topics and describes them. Documents and words inhibit the same latent semantic space, whose dimensions are the topics.

The implementation is based on a lightweight neural architecture and aims to be a scalable alternative to LDA. It readily makes use of GPU computation and has been tested successfully on 1M documents with 100 topics.

Getting started: python -m tests.basic.py data/my_docs.txt
 
