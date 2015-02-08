import re
import cPickle as pickle

def isurl(token):
	return True if re.search("((mailto\:|(news|(ht|f)tp(s?))\://){1}\S+)",token) else False

def isusername(token):
	return True if re.search("@+[\w_]+",token) else False

def ishashtag(token):
	return True if re.search("\#+[\w_]+[\w\'_\-]*[\w_]+",token) else False

def isnumber(token):
	return True if re.search("[\d]+",token) else False

def replace(tokenized):
	replaced_tokenized = list()
	for sample in tokenized:
		tokens = list()
		for token in sample:
			token = token.lower()
			if isusername(token):
				token = "@mention"
			elif ishashtag(token):
				token = "#hash"
			elif isurl(token):
				token = "$url$"
			elif isnumber(token):
				token = "$num$"
			tokens.append(token)
		replaced_tokenized.append(tokens)
	return replaced_tokenized

train_tokenized = pickle.load(open("../data/intermediate/train_tokenized.pkl","r"))
test_tokenized = pickle.load(open("../data/intermediate/test_tokenized.pkl","r"))

train_tokenized = replace(train_tokenized)
test_tokenized = replace(test_tokenized)

pickle.dump(train_tokenized,open("../data/intermediate/train_tokenized_replaced.pkl","w"))
pickle.dump(test_tokenized,open("../data/intermediate/test_tokenized_replaced.pkl","w"))
