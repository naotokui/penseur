import numpy as np
import cPickle as pickle
import os, skipthoughts

class Penseur:

	def __init__(self):
		self.model = skipthoughts.load_model()
		self.current_text = None
		self.vectors = None
		self.analogy_vector = None

	def load(self, filename):
		self.vectors = np.load(filename + '_enc.np', 'r')
		self.current_text = pickle.load(open(filename + '_sen.p', 'r'))

	def encode(self, sentences):
		self.current_text = sentences
		self.vectors = skipthoughts.encode(self.model, sentences)

	def save(self, filename):
		np.save(open(filename + '_enc.np', 'w'), self.vectors)
		pickle.dump(self.current_text, open(filename + '_sen.p', 'w'))

	def get_closest_sentences(self, query_sentence, num_results=5):
		return skipthoughts.nn(self.model, self.current_text, self.vectors, query_sentence, num_results)

	def get_vector(self, query_sentence):
		return skipthoughts.vector(self.model, self.current_text, self.vectors, query_sentence)

	def get_sentence(self, query_vector):
		return skipthoughts.sentence(self.model, self.current_text, self.vectors, query_vector)

	def load_pairs(self, filename):
		with open(filename + '.txt', 'r') as f:
			s = f.readlines()
		av = []
		for i in xrange(0, len(s), 3):
			cv = self.get_vector(s[i+1].replace('\n', '')) - self.get_vector(s[i].replace('\n', ''))
			av.append(cv)
		return np.average(np.array(av), axis=0)

	def analogy(self, query_sentence, filename='q&a_pairs'):
		if self.analogy_vector is None:
			if os.path.isfile(filename + '.np'):
				self.analogy_vector = np.load(filename + '.np', 'r')
			else:
				self.analogy_vector = self.load_pairs(filename)
				np.save(open(filename + '.np', 'w'), self.analogy_vector)
		return self.get_sentence(self.get_vector(query_sentence) + self.analogy_vector)

