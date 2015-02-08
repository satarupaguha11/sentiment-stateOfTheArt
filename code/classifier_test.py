import config
from helper import *
from sklearn.svm import LinearSVC
from generate_features import *
from evaluate import *

def main():
	
	word_ngrams = config.word_ngrams
	if word_ngrams == True:
		vocab = load_pickle("../data/intermediate/unigram_dict.pkl")
	else:
		vocab = load_pickle("../data/intermediate/unigram_dict.pkl")
	test_features = generate_features('test',vocab,word_ngrams)
	test_labels = load_pickle("../data/intermediate/test_labels.pkl")
	print "Feature matrix dimensions: "+str(test_features.shape)
	clf = load_pickle("../data/intermediate/classifier.model")
	predicted_labels = clf.predict(test_features)
	print predicted_labels
	confusion_matrix = evaluate(test_labels,predicted_labels)
	display_confusion_matrix(confusion_matrix)

if __name__=="__main__":
	main()