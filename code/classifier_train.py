import config
from helper import *
from sklearn.svm import LinearSVC
from generate_features import *

def main():
	
	word_ngrams = config.word_ngrams
	n=config.n
	if word_ngrams == True:
		if n==1:
			vocab = load_pickle("../data/intermediate/unigram_dict.pkl")
		elif n==2:
			vocab = load_pickle("../data/intermediate/bigram_dict.pkl")
		elif n==3:
			vocab = load_pickle("../data/intermediate/trigram_dict.pkl")
		elif n==4:
			vocab = load_pickle("../data/intermediate/fourgram_dict.pkl")
	train_features = generate_features('train',vocab,n,word_ngrams)
	train_labels = load_pickle("../data/intermediate/train_labels.pkl")
	print "Feature matrix dimensions: "+str(train_features.shape)
	clf = LinearSVC(C=config.C, class_weight='auto')
	clf.fit(train_features,train_labels)
	save_pickle(clf,"../data/intermediate/classifier.model")

if __name__=="__main__":
	main()