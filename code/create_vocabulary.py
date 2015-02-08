import cPickle as pickle
import re
from collections import defaultdict


def main():
	train_tokenized = pickle.load(open("../data/intermediate/train_tokenized_replaced.pkl","r"))
	test_tokenized = pickle.load(open("../data/intermediate/test_tokenized_replaced.pkl","r"))
	tokenized = train_tokenized+test_tokenized
	index = 0
	unigram_dict,index = create_ngram_dict(index,tokenized,1) 
	bigram_dict,index = create_ngram_dict(index,tokenized,2)
	bigram_dict = dict(unigram_dict.items()+bigram_dict.items())
	trigram_dict,index = create_ngram_dict(index, tokenized,3)
	trigram_dict= dict(bigram_dict.items()+trigram_dict.items())
	fourgram_dict,index = create_ngram_dict(index, tokenized,4)
	fourgram_dict= dict(trigram_dict.items()+fourgram_dict.items())
	pickle.dump(unigram_dict,open("../data/intermediate/unigram_dict.pkl","w"))
	pickle.dump(bigram_dict,open("../data/intermediate/bigram_dict.pkl","w"))
	pickle.dump(trigram_dict,open("../data/intermediate/trigram_dict.pkl","w"))
	pickle.dump(fourgram_dict,open("../data/intermediate/fourgram_dict.pkl","w"))

def create_ngram_dict(index,tokenized,n):
	from nltk.util import ngrams
	"""
	n_grams = defaultdict()
	index = 0
	for sample in tokenized:
		for token_no in range(len(sample)-n):
			string = sample[token_no]
			for i in range(1,n):
				string += ' '+sample[token_no+i]
			if string not in n_grams.keys():
				print string
				n_grams[string] = index
	print len(n_grams)
	"""
	dictionary = defaultdict()
	count = 0
	for sample in tokenized:
		count+=1
		temp = ngrams(sample, n)
		for i in temp:
			i = i.lower()
			if i not in dictionary.keys():
				dictionary[i] = index
				index+=1
		#print count
	return dictionary,index

if __name__ == "__main__":
	main()