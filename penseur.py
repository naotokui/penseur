import sPickle as pickle
import os, skipthoughts

class Penseur:

	def __init__(self):
		self.model = skipthoughts.load_model()

	def load_encodings(self, filename):
		self.vectors = pickle.s_load(open(filename + '.spkl', 'r'))

	def encode_and_save(self, sentences, name):
		v = skipthoughts.encode(self.model, sentences)
		pickle.s_dump( , open(name + '.spkl', 'r'))

