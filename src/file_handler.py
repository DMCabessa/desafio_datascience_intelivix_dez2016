from nltk.corpus import brown
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
import string

def readfile(fileid):
	rawfile = brown.open(fileid)
	return rawfile.read()

def preprocess(file):
	words = file.split()
	_stopwords = stopwords.words('english')
	stemmer = LancasterStemmer()
	expanded_punctuation = ""
	for symbol in string.punctuation:
		expanded_punctuation += symbol+symbol

	# Remove the tags and punctuation from text
	untagged = []
	for word in words:
		coreword = word[:word.index('/')]
		if coreword not in expanded_punctuation:
			untagged.append(coreword)

	# Remove stopwords from text
	filtered = [w for w in untagged if w.lower() not in _stopwords]

	# Stemm text
	for i in range(0,len(filtered)):
		filtered[i] = stemmer.stem(filtered[i])

	return filtered


def getrelevant():
	cat0 = u'learned' 			# cjNN
	cat1 = u'belles_lettres'	# cgNN

	_id = u'cats.txt'
	file = brown.open(_id)

	relevant_files = []
	for line in file:
		category = line.rstrip('\n').split()[1]
		if category == cat0 or category == cat1:
			relevant_files.append(line.split()[0])

	return relevant_files