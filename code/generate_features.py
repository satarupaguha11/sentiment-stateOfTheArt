from helper import *
from nltk.util import ngrams
from numpy import *
bingliu_lexicon_dictionary = pickle.load(open('../data/lexicons/bingliu_lexicon_dictionary.pkl','r'))

def compute_n_gram(sample,vocab,n):
	feature_vector = zeros((len(vocab)),dtype="int32")
	tokens = ngrams(sample,n)
	#print tokens
	for token in tokens:
		if token in vocab:
			feature_vector[vocab[token]] = 1
	return feature_vector
def compute_manual_lexicons(sample):
	no_of_words_in_sentence = len(sample)
	feature_vector = zeros((2),dtype="int32")
	num_pos_tokens = 0
	num_neg_tokens = 0
	for unigram in sample:
		unigram = unigram.lower()
		if unigram in bingliu_lexicon_dictionary['positive']:
			num_pos_tokens+=1
		elif unigram in bingliu_lexicon_dictionary['negative']:
			num_neg_tokens+=1
	feature_vector[0]=num_pos_tokens
	feature_vector[1]=num_neg_tokens
	return feature_vector
def generate_features(split,n,word_ngrams,manual_lexicons):
	n_features = 0
	tokenized = load_pickle("../data/intermediate/"+split+"_tokenized_replaced.pkl")
	n_samples = len(tokenized)
	if word_ngrams == True:
		if n==1:
			vocab = load_pickle("../data/intermediate/unigram_dict.pkl")
		elif n==2:
			vocab = load_pickle("../data/intermediate/bigram_dict.pkl")
		elif n==3:
			vocab = load_pickle("../data/intermediate/trigram_dict.pkl")
		elif n==4:
			vocab = load_pickle("../data/intermediate/fourgram_dict.pkl")
		n_features+=len(vocab)
	if manual_lexicons == True:
		n_features+=2
	
	feature_matrix = zeros((n_samples,n_features),dtype="int32")
	
	for sample_no in range(len(tokenized)):
		sample = tokenized[sample_no]
		#print sample
		if word_ngrams == True:
			ngram_features = compute_n_gram(sample,vocab,n)
		if manual_lexicons == True:
			manual_lexicon_features = compute_manual_lexicons(sample)
		feature_matrix[sample_no,:] = concatenate([ngram_features,manual_lexicon_features])
	return feature_matrix