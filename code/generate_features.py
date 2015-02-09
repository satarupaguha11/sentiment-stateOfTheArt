from helper import *
from nltk.util import ngrams
from numpy import *
import re
bingliu_lexicon_dictionary = pickle.load(open('../data/lexicons/bingliu_lexicon_dictionary.pkl','r'))
sentiment140_lexicon_dictionary = pickle.load(open('../data/lexicons/sentiment140_lexicon_dictionary.pkl','r'))
sentiment_hashtag_lexicon_dictionary = pickle.load(open('../data/lexicons/sentiment_hashtag_lexicon_dictionary.pkl','r'))
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

def compute_pmi_unigram_lexicons(sample):
	feature_vector = zeros((12),dtype="int32")
	no_of_words_in_sentence = len(sample)
	scoreSum = 0
	scores=[0]
	num_pos = 0
	num_neg = 0
	last_score = 0
	score = 0

	for unigram in sample:
	
		if unigram in sentiment140_lexicon_dictionary:
			
			score = sentiment140_lexicon_dictionary[unigram]
			
			if score > 1:
				num_pos += 1
			elif score < -1:
				num_neg += 1
			scoreSum+=score
			scores.append(score)
	maxScore = max(scores)
	minScore = min(scores)
	last_score = score
	
	feature_vector[0]=scoreSum#/float(no_of_words_in_sentence)
	feature_vector[1]=maxScore#/float(no_of_words_in_sentence)
	feature_vector[2]=minScore#/float(no_of_words_in_sentence)
	feature_vector[3]=num_pos#/float(no_of_words_in_sentence)
	feature_vector[4]=num_neg#/float(no_of_words_in_sentence)
	feature_vector[5]=last_score#/float(no_of_words_in_sentence)
	scoreSum = 0
	scores=[0]
	num_pos = 0
	num_neg = 0
	last_score = 0
	score = 0

	for unigram in sample:
		if unigram in sentiment_hashtag_lexicon_dictionary:
			
			score = sentiment_hashtag_lexicon_dictionary[unigram]
			if score > 1:
				num_pos += 1
			elif score < -1:
				num_neg += 1
			scoreSum+=score
			scores.append(score)
	maxScore = max(scores)
	minScore = min(scores)
	last_score = score
	feature_vector[6]=scoreSum#/float(no_of_words_in_sentence)
	feature_vector[7]=maxScore#/float(no_of_words_in_sentence)
	feature_vector[8]=minScore#/float(no_of_words_in_sentence)
	feature_vector[9]=num_pos#/float(no_of_words_in_sentence)
	feature_vector[10]=num_neg#/float(no_of_words_in_sentence)
	feature_vector[11]=last_score#/float(no_of_words_in_sentence)
	return feature_vector
def compute_hashtag_features(sample):
	feature_vector = zeros((1),dtype="int32")
	num = 0
	no_of_words_in_sentence = len(sample)
	for token in sample:
		if token == "#hash":
			#print "yay"
			num +=1
	feature_vector[0] = num#/float(no_of_words_in_sentence)
	return feature_vector

def compute_grapheme_features(sample):
	feature_vector = zeros((1),dtype="int32")
	num = 0
	for token in sample:
		repeatMatch = re.search(r"\w+((\w)\2\2)\w*", token)
		if repeatMatch:
			num+=1
	feature_vector[0] = num
	return feature_vector
def compute_negation_features(sample):
	feature_vector = zeros((2),dtype="int32")
	start = -1;end = -1
	for token_no in range(len(sample)):
		if re.search("(?:^(?:never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint)$)|n't",sample[token_no]):
			start = token_no
		if re.search("^[.:;!?]$",sample[token_no]):
			end = token_no
	if start!=-1:
		negated_context = sample[start:end+1]
		num_pos = 0; num_neg = 0;
		for item in negated_context:
			if item in bingliu_lexicon_dictionary['positive']:
				num_neg+=1
			elif item in bingliu_lexicon_dictionary['negative']:
				num_pos+=1
		feature_vector[0] = num_pos
		feature_vector[1] = num_neg
	return feature_vector

def generate_features(split,n,word_ngrams,manual_lexicons,emoticons,last_senti_word,pmi_unigram_lexicons,hashtag,grapheme,negation):
	n_features = 0
	tokenized = load_pickle("../data/intermediate/"+split+"_tokenized_replaced.pkl")
	n_samples = len(tokenized)
	emoticon_features = array([])
	manual_lexicon_features = array([])
	ngram_features = array([])
	last_senti_word_features = array([])
	pmi_unigram_lexicon_features = array([])
	hashtag_features = array([])
	grapheme_features = array([])
	negation_features = array([])
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
	if pmi_unigram_lexicons == True:
		n_features += 12
	if hashtag == True:
		n_features +=1
	if grapheme == True:
		n_features += 1
	if negation == True:
		n_features += 2
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
		if pmi_unigram_lexicons == True:
			pmi_unigram_lexicon_features = compute_pmi_unigram_lexicons(sample)
		if hashtag == True:
			hashtag_features = compute_hashtag_features(sample)
		if grapheme == True:
			grapheme_features = compute_grapheme_features(sample)
		if negation == True:
			negation_features = compute_negation_features(sample)
		feature_matrix[sample_no,:] = concatenate([ngram_features,manual_lexicon_features,emoticon_features,last_senti_word_features,pmi_unigram_lexicon_features,hashtag_features,grapheme_features,negation_features])
	return feature_matrix