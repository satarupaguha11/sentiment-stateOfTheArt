import config
from helper import *
from sklearn.svm import LinearSVC
from generate_features import *

def main():
	
	word_ngrams = config.word_ngrams
	manual_lexicons = config.manual_lexicons
	n=config.n
	train_features = generate_features('train',n,word_ngrams,manual_lexicons)
	train_labels = load_pickle("../data/intermediate/train_labels.pkl")
	print "Feature matrix dimensions: "+str(train_features.shape)
	clf = LinearSVC(C=config.C, class_weight='auto')
	clf.fit(train_features,train_labels)
	save_pickle(clf,"../data/intermediate/classifier.model")

if __name__=="__main__":
	main()