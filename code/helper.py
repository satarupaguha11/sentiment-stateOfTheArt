import cPickle as pickle
def load_pickle(loadPath):

	model = pickle.load(open(loadPath,'r'))
	return model

def save_pickle(model,dumpPath):
	
	pickle.dump(model,open(dumpPath,'w'))
