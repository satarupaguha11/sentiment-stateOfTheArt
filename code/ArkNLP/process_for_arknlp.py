f_train = open("../../data/input/train.txt")
f_test = open("../../data/input/test.txt")

fout_train = open("../../data/intermediate/train_text.txt","w")
for line in f_train:
	text = line.strip().split("\t")[-1]
	fout_train.write(text+"\n")
fout_train.close()


fout_test = open("../../data/intermediate/test_text.txt","w")
for line in f_test:
	text = line.strip().split("\t")[-1]
	fout_test.write(text+"\n")
fout_test.close()

