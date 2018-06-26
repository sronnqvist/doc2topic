from doc2topic import models, corpora
import sys

data = corpora.DocData(sys.argv[1])
model = models.Doc2Topic(data, n_topics=30)

model.print_topic_words()
