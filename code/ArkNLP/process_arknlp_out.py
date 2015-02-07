import cPickle as pickle

fin_train = open("../../data/intermediate/train_arknlp.txt")
fin_test = open("../../data/intermediate/test_arknlp.txt")

train_tokenized = list()
train_pos = list()

for line in fin_train:
	fields = line.strip().split("\t")
	tokens = fields[0].split()
	pos_tags = fields[1].split()
	train_tokenized.append(tokens)
	train_pos.append(pos_tags)

test_tokenized = list()
test_pos = list()

for line in fin_test:
	fields = line.strip().split("\t")
	tokens = fields[0].split()
	pos_tags = fields[1].split()
	test_tokenized.append(tokens)
	test_pos.append(pos_tags)

pickle.dump(train_tokenized,open("../../data/intermediate/train_tokenized.pkl","w"))
pickle.dump(test_tokenized,open("../../data/intermediate/test_tokenized.pkl","w"))
pickle.dump(train_pos,open("../../data/intermediate/train_pos.pkl","w"))
pickle.dump(test_pos,open("../../data/intermediate/test_pos.pkl","w"))

