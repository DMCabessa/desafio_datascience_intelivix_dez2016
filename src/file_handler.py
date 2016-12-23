import itertools
import numpy as np
import matplotlib.pyplot as plt

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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')