import cPickle as pickle

f_train = open("../../data/input/train.txt", "r")
f_test = open("../../data/input/test13.txt", "r")
train_labels = list()
test_labels = list()

encodeLabel = {"positive":1,"negative":-1,"neutral":0}

fout_train = open("../../data/intermediate/train_text.txt","w")
for line in f_train:
	fields = line.strip().split("\t")
	text = fields[-1]
	label = fields[2]
	fout_train.write(text+"\n")
	train_labels.append(encodeLabel[label])
fout_train.close()

fout_test = open("../../data/intermediate/test_text.txt","w")
for line in f_test:
	fields = line.strip().split("\t")
	text = fields[-1]
	label = fields[2]
	fout_test.write(text+"\n")
	test_labels.append(encodeLabel[label])
fout_test.close()

pickle.dump(train_labels,open("../../data/intermediate/train_labels.pkl","w"))
pickle.dump(test_labels,open("../../data/intermediate/test_labels.pkl","w"))
