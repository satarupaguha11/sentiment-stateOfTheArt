from helper import *
from nltk.util import ngrams
from numpy import *

def compute_n_gram(sample,vocab,n):
	feature_vector = zeros((len(vocab)),dtype="int32")
	tokens = ngrams(sample,n)
	#print tokens
	for token in tokens:
		if token in vocab:
			feature_vector[vocab[token]] = 1
	return feature_vector

def generate_features(split,n,word_ngrams):
	tokenized = load_pickle("../data/intermediate/"+split+"_tokenized_replaced.pkl")
	if word_ngrams == True:
		if n==1:
			vocab = load_pickle("../data/intermediate/unigram_dict.pkl")
		elif n==2:
			vocab = load_pickle("../data/intermediate/bigram_dict.pkl")
		elif n==3:
			vocab = load_pickle("../data/intermediate/trigram_dict.pkl")
		elif n==4:
			vocab = load_pickle("../data/intermediate/fourgram_dict.pkl")

	n_samples = len(tokenized)
	n_features = 0
	if word_ngrams == True:
		n_features+=len(vocab)
	feature_matrix = zeros((n_samples,n_features),dtype="int32")
	
	for sample_no in range(len(tokenized)):
		sample = tokenized[sample_no]
		#print sample
		if word_ngrams == True:
			ngram_features = compute_n_gram(sample,vocab,n)
		feature_matrix[sample_no,:] = ngram_features
	return feature_matrix