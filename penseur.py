import numpy as np
import sPickle as pickle
import os, skipthoughts

class Penseur:

	def __init__(self):
		self.model = skipthoughts.load_model()
		self.current_text = None
		self.vectors = None
		self.num_sentences = 0

	def load(self, filename):
		self.vectors_tmp, self.current_text_tmp, self.num_sentences = pickle.s_load(open(filename + '_enc.spkl', 'r'))
		self.vectors = np.array( np.empty((self.num_sentences, 4800)), dtype=object )
		for i, el in enumerate(self.vectors_tmp):
			self.vectors[i] = el
		self.current_text = np.array( np.empty((self.num_sentences, 1)), dtype=object )
		for i, el in enumerate(self.current_text_tmp):
			self.current_text[i] = el

	def encode(self, sentences):
		self.current_text = sentences
		self.vectors = skipthoughts.encode(self.model, sentences)
		self.num_sentences = len(sentences)

	def save(self, name):
		f = [self.vectors, self.current_text, self.num_sentences]
		pickle.s_dump(f, open(name + '_enc.spkl', 'w'))

	def get_closest_sentences(self, query_sentence, num_results=5):
		return skipthoughts.nn(self.model, self.current_text, self.vectors, query_sentence, num_results)

	def get_vector(self, query_sentence):
		return skipthoughts.vector(self.model, self.current_text, self.vectors, query_sentence)

	def get_sentence(self, query_vector):
		return skipthoughts.sentence(self.model, self.current_text, self.vectors, query_vector)

	def analogy(self, query_sentence):
		with open('test_q&a.txt', 'r') as f:
			s = f.readlines()
		av = []
		for i in xrange(0, len(s), 3):
			cv = self.get_vector(s[i+1].replace('\n', '')) - self.get_vector(s[i].replace('\n', ''))
			av.append(cv)
		new_av = np.average(np.array(av), axis=0)
		return self.get_sentence(self.get_vector(query_sentence) + new_av)

