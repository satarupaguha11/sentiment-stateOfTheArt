import config
from helper import *
from sklearn.svm import LinearSVC
from generate_features import *
from evaluate import *

def main():
	
	word_ngrams = config.word_ngrams
	manual_lexicons = config.manual_lexicons
	n=config.n
	emoticons = config.emoticons
	last_senti_word = config.last_senti_word
	pmi_unigram_lexicons = config.pmi_unigram_lexicons
	hashtag = config.hashtag
	grapheme = config.grapheme
	negation = config.negation
	test_features = generate_features('test',n,word_ngrams,manual_lexicons,emoticons,last_senti_word,pmi_unigram_lexicons,hashtag,grapheme,negation)
	test_labels = load_pickle("../data/intermediate/test_labels.pkl")
	print "Feature matrix dimensions: "+str(test_features.shape)
	clf = load_pickle("../data/intermediate/classifier.model")
	predicted_labels = clf.predict(test_features)
	confusion_matrix = evaluate(test_labels,predicted_labels)
	display_confusion_matrix(confusion_matrix)

if __name__=="__main__":
	main()