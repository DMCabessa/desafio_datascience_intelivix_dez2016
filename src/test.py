from __future__ import division

import random
import file_handler
import csv
import numpy

from inverted_index import InvertedIndex
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn import svm

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

test_range = 10

# Loop for measuring avg mse
mse_dict = {'mlp':list(), 'tree':list(), 'svm':list()}

for gen in range(1,(test_range+1)):
	print "\nRunning test loop "+str(gen)+" out of "+str(test_range)+"..."

	relevant_files = file_handler.getrelevant()
	trainset = random.sample(relevant_files, 103)
	testset = [x for x in relevant_files if x not in trainset]

	index = InvertedIndex(trainset)
	index.trim()
	training_samples, training_classes = index.tf_idf()
	test_samples = index.tf_idf(testset)

	true_classes = []
	for fileid in testset:
		_class = 0 if fileid[1] == 'j' else 1
		true_classes.append(_class)

	# ==================================================
	# MLP classifier
	# ==================================================

	# Training step
	#print "Starting training step..."
	classifier = MLPClassifier()
	classifier.fit(training_samples, training_classes)
	#print "Finished training."

	# Testing step
	#print "Starting testing step..."
	predicted_classes = classifier.predict(test_samples)

	misses = 0
	for i in range(0, len(testset)):
		if true_classes[i] != predicted_classes[i]:
			misses += 1

	mse = misses / len(testset)
	print "MLP::mse: \t\t"+str(mse)
	mse_dict['mlp'].append(mse)

	# Confusion matrix setop
	cnf_matrix = confusion_matrix(true_classes, predicted_classes)
	numpy.set_printoptions(precision=2)

	plt.figure()
	file_handler.plot_confusion_matrix(cnf_matrix, classes=['learned','belles_letters'],
		title='MLP confusion matrix')

	plt.savefig('../results/confusion-matrix/'+str(gen)+'-mlp-confusion-matrix.png')

	# ==================================================
	# Decision tree classifier
	# ==================================================

	# Training step
	#print "Starting training step..."
	classifier = tree.DecisionTreeClassifier()
	classifier.fit(training_samples, training_classes)
	#print "Finished training."

	# Testing step
	#print "Starting testing step..."
	predicted_classes = classifier.predict(test_samples)

	misses = 0
	for i in range(0, len(testset)):
		if true_classes[i] != predicted_classes[i]:
			misses += 1

	mse = misses / len(testset)
	print "Decision tree::mse: \t"+str(mse)
	mse_dict['tree'].append(mse)

	# Confusion matrix setop
	cnf_matrix = confusion_matrix(true_classes, predicted_classes)
	numpy.set_printoptions(precision=2)

	plt.figure()
	file_handler.plot_confusion_matrix(cnf_matrix, classes=['learned','belles_letters'],
		title='Decision tree confusion matrix')

	plt.savefig('../results/confusion-matrix/'+str(gen)+'-tree-confusion-matrix.png')

	# ==================================================
	# Support vector machine classifier
	# ==================================================

	# Training step
	#print "Starting training step..."
	classifier = svm.SVC()
	classifier.fit(training_samples, training_classes)
	#print "Finished training."

	# Testing step
	#print "Starting testing step..."
	predicted_classes = classifier.predict(test_samples)

	misses = 0
	for i in range(0, len(testset)):
		if true_classes[i] != predicted_classes[i]:
			misses += 1

	mse = misses / len(testset)
	print "SVM::mse: \t\t"+str(mse)
	mse_dict['svm'].append(mse)

	# Confusion matrix setop
	cnf_matrix = confusion_matrix(true_classes, predicted_classes)
	numpy.set_printoptions(precision=2)

	plt.figure()
	file_handler.plot_confusion_matrix(cnf_matrix, classes=['learned','belles_letters'],
		title='SVM confusion matrix')

	plt.savefig('../results/confusion-matrix/'+str(gen)+'-svm-confusion-matrix.png')

print "\n\nAverage mse: "
print "\tMLP -> \t\t"+str(numpy.mean(mse_dict['mlp']))
print "\tDecision tree -> \t"+str(numpy.mean(mse_dict['tree']))
print "\tSVM -> \t\t"+str(numpy.mean(mse_dict['svm']))