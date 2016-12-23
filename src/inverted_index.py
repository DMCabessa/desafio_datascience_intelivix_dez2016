import json
import time
import math
import file_handler

class InvertedIndex():

	def __init__(self, trainset):
		self.table = {}
		self.tokens = []
		self.documents = trainset

		start = time.time()

		for fileid in trainset:
			file = file_handler.readfile(fileid)
			processed = file_handler.preprocess(file)

			for token in processed:

				if token not in self.table:
					self.table[token] = dict()
					self.tokens.append(token)

				token_data = self.table[token]
				if fileid not in token_data:
					token_data[fileid] = 1
				else:
					token_data[fileid] += 1

		finish = time.time()
		elapsed = finish - start
		print "Built inverted index from "+str(len(self.documents))+" documents in "+str(elapsed)+" seconds."

	def __str__(self):
		return json.dumps(self.table, indent=1)

	def __repr__(self):
		return json.dumps(self.table, indent=1)

	def tf_idf(self, idset=[]):
		if len(idset) == 0:
			start = time.time()
			matrix = [[0 for x in range(len(self.tokens))] for y in range(len(self.documents))]
			classes = []

			step = 100.0/len(self.documents)
			for i in range(0, len(self.documents)):
				for j in range(0,len(self.tokens)+1):
					if j == len(self.tokens):
						if self.documents[i][1] == 'j':
							classes.append(0)
						else:
							classes.append(1)
					else:
						tf = self._tf(self.tokens[j], self.documents[i])
						idf = self._idf(self.tokens[j])

						matrix[i][j] = tf*idf

				val = i*step
				print ""+str(val)+"% complete..."

			finish = time.time()
			elapsed = finish - start
			print "Built tf-idf matrix in "+str(elapsed)+" seconds."

			return matrix, classes

		else:
			start = time.time()
			matrix = [[0 for x in range(len(self.tokens))] for y in range(len(idset))]

			step = 100.0/len(idset)
			for i in range(0, len(idset)):
				file = file_handler.readfile(idset[i])
				processed = file_handler.preprocess(file)

				for j in range(0,len(self.tokens)):
					tf = self._tf(self.tokens[j], processed)
					idf = self._idf(self.tokens[j])

					matrix[i][j] = tf*idf

				val = i*step
				print ""+str(val)+"% complete..."

			finish = time.time()
			elapsed = finish - start
			print "Built tf-idf matrix in "+str(elapsed)+" seconds."

			return matrix
		

	def _tf(self, term, document):
		if isinstance(document, list):
			if term in document:
				frequency = 0
				for token in document:
					if token == term:
						frequency += 1
				return 1 + math.log10(frequency)
			else:
				return 0
		else:
			if document in self.table[term]:
				frequency = self.table[term][document]
				return 1 + math.log10(frequency)
			else:
				return 0

	def _idf(self, term):
		N = len(self.tokens)
		document_frequency = len(self.table[term].keys())
		return math.log10(N/1 + document_frequency)