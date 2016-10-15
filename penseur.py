import sPickle as pickle
import os, skipthoughts

class Penseur:

	def __init__(self):
		self.model = skipthoughts.load_model()
		self.current_text = None

	def load_encodings(self, filename):
		self.vectors = pickle.s_load(open(filename + '.spkl', 'r'))
		self.current_text = 

	def encode_and_save(self, sentences, name):
		v = skipthoughts.encode(self.model, sentences)
		pickle.s_dump(self.current_text, open())
		pickle.s_dump( , open(name + '.spkl', 'r'))

	def nn(self, query_sentence, num_results=5):
		return skipthoughts.nn(self.model, )
