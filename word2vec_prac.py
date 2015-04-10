__author__ = 'zhangye'
from gensim.models.word2vec import Word2Vec
print("load model")
model = Word2Vec.load('bio_word',mmap='r')
#model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary = True)
#model = Word2Vec.load_word2vec_format('wikipedia-pubmed-and-PMC-w2v.bin',binary=True)
#model.init_sims(replace=True)
#model.save('bio_word')
print("finish load model")
#model = Word2Vec.load("bio_word")
print model.similarity('men','man')
