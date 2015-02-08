from helper import *
from nltk.util import ngrams
from numpy import *
import re
bingliu_lexicon_dictionary = pickle.load(open('../data/lexicons/bingliu_lexicon_dictionary.pkl','r'))
_regex_emoticon_happy = re.compile(r"((:)+(-)*(\))+)|((\()+(-)*(:)+)|<3|XD")
_regex_emoticon_sad = re.compile(r"((:)+(-)*(\()+)")

def compute_n_gram(sample,vocab,n):
	feature_vector = zeros((len(vocab)),dtype="int32")
	tokens = ngrams(sample,n)
	#print tokens
	for token in tokens:
		#if token in vocab:
			feature_vector[vocab[token]] = 1
	return feature_vector
def compute_manual_lexicons(sample):
	no_of_words_in_sentence = len(sample)
	feature_vector = zeros((2),dtype="int32")
	num_pos_tokens = 0
	num_neg_tokens = 0
	for unigram in sample:
		#unigram = unigram.lower()
		if unigram in bingliu_lexicon_dictionary['positive']:
			num_pos_tokens+=1
		elif unigram in bingliu_lexicon_dictionary['negative']:
			num_neg_tokens+=1
	feature_vector[0]=num_pos_tokens
	feature_vector[1]=num_neg_tokens
	return feature_vector
def compute_emoticon_features(sample):
	feature_vector = zeros((2),dtype="int32")
	for token in sample:
		if re.search(_regex_emoticon_happy, token):
			#print token
			feature_vector[0] = 1
		elif re.search(_regex_emoticon_sad, token):
			#print token
			feature_vector[1] = 1
	return feature_vector
def compute_last_senti_word(sample):
	feature_vector = zeros((1),dtype="int32")
	flag = -1
	for token in sample:
		if token in bingliu_lexicon_dictionary['positive']:
			flag = 0
		elif token in bingliu_lexicon_dictionary['negative']:
			flag = 1
	if flag == 0:
		feature_vector[0] = 1
	elif flag == 1:
		feature_vector[0] = -1
	return feature_vector
	
def generate_features(split,n,word_ngrams,manual_lexicons,emoticons,last_senti_word):
	n_features = 0
	tokenized = load_pickle("../data/intermediate/"+split+"_tokenized_replaced.pkl")
	n_samples = len(tokenized)
	emoticon_features = array([])
	manual_lexicon_features = array([])
	ngram_features = array([])
	last_senti_word_features = array([])
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
	if emoticons == True:
		n_features+=2
	if last_senti_word == True:
		n_features +=1
	
	feature_matrix = zeros((n_samples,n_features),dtype="int32")
	
	for sample_no in range(len(tokenized)):
		sample = tokenized[sample_no]
		#print sample
		if word_ngrams == True:
			ngram_features = compute_n_gram(sample,vocab,n)
		if manual_lexicons == True:
			manual_lexicon_features = compute_manual_lexicons(sample)
		if emoticons == True:
			emoticon_features = compute_emoticon_features(sample)
		if last_senti_word == True:
			last_senti_word_features = compute_last_senti_word(sample)
		feature_matrix[sample_no,:] = concatenate([ngram_features,manual_lexicon_features,emoticon_features,last_senti_word_features])
	return feature_matrix