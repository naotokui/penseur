import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import os, skipthoughts

class Penseur:

	def __init__(self):
		self.model = skipthoughts.load_model()
		self.sentences = None
		self.vectors = None
		self.analogy_vector = None

	# Loads both an encoding file and its sentences from disc
	def load(self, filename):
		self.vectors = np.load(filename + '_enc.np', 'r')
		self.sentences = pickle.load(open(filename + '_sen.p', 'r'))

	# Encodes a list of sentences
	def encode(self, sentences):
		self.sentences = sentences
		self.vectors = skipthoughts.encode(self.model, sentences)

	# Saves a set of encodings and the corresponding sentences to disc
	def save(self, filename):
		np.save(open(filename + '_enc.np', 'w'), self.vectors)
		pickle.dump(self.sentences, open(filename + '_sen.p', 'w'))

	# Returns a list of the sentences closest to the input
	def get_closest_sentences(self, query_sentence, num_results=5):
		return skipthoughts.nn(self.model, self.sentences, self.vectors, query_sentence, num_results)

	# Returns the vector of a query sentence within the current embedding space
	def get_vector(self, query_sentence):
		return skipthoughts.vector(self.model, self.sentences, self.vectors, query_sentence)

	# Returns the sentence of a query vector
	def get_sentence(self, query_vector):
		return skipthoughts.sentence(self.model, self.sentences, self.vectors, query_vector)

	# Loads pairs of sentences (ie questions and answers) from disc
	def load_pairs(self, filename):
		with open(filename + '.txt', 'r') as f:
			s = f.readlines()
		av = []
		for i in xrange(0, len(s), 3):
			cv = self.get_vector(s[i+1].replace('\n', '')) - self.get_vector(s[i].replace('\n', ''))
			av.append(cv)
		return np.average(np.array(av), axis=0)

	# Returns the response using the average vector from load_pairs input file
	def analogy(self, query_sentence, filename='q&a_pairs'):
		if self.analogy_vector is None:
			if os.path.isfile(filename + '.np'):
				self.analogy_vector = np.load(filename + '.np', 'r')
			else:
				self.analogy_vector = self.load_pairs(filename)
				np.save(open(filename + '.np', 'w'), self.analogy_vector)
		return self.get_sentence(self.get_vector(query_sentence) + self.analogy_vector)

	# Displays the plot of the sentence encodings after PCA (to 2D)
	def display_PCA_plot(self):
		try:
			plot_data = self.PCA(np.squeeze(np.array(self.vectors)))
			for i, v in enumerate(plot_data):
				plt.scatter(v[0], v[1])
				plt.annotate(self.sentences[i], (v[0], v[1]))
			plt.title("PCA plot")
			plt.show()
		except:
			print("Not enough Memory; corpus too large for this function")

	# Performs PCA on the sentence encodings
	def PCA(self, data, rescaled_dims=2):
		m, n = data.shape

		# Center around the mean
		plot_data = data - data.mean(axis=0)

		# Covariance matrix
		r = np.cov(plot_data, rowvar=False)

		# Get eigenvals, eigenvectors
		evals, evecs = np.linalg.eigh(r)

		# Sort eigevalue decreasing order
		idx = np.argsort(evals)[::-1]
		evecs = evecs[:,idx]

		# Sort eigenvects by same index
		evals = evals[idx]

		# Select first n eigenvectors
		evecs = evecs[:, :rescaled_dims]

		return np.dot(evecs.T, plot_data.T).T

