'''
Created by Satarupa Guha on 21 Jan 2015
Expected Output: Confusion matrix, i.e. precision, recall and f-measure for each of the classes.
'''
from sklearn.metrics import precision_recall_fscore_support

def evaluate(y_true, y_pred):
	return precision_recall_fscore_support(y_true, y_pred)
	
def display_confusion_matrix(confusion_matrix):
	print "The confusion matrix is"
	print "         \tNegative     \tPositive     \tNeutral"
	print "Precision\t"+str(confusion_matrix[0][0])+'\t'+str(confusion_matrix[0][1])+'\t'+str(confusion_matrix[0][2])
	print "Recall   \t"+str(confusion_matrix[1][0])+'\t'+str(confusion_matrix[1][1])+'\t'+str(confusion_matrix[1][2])
	print "F-measure\t"+str(confusion_matrix[2][0])+'\t'+str(confusion_matrix[2][1])+'\t'+str(confusion_matrix[2][2])
