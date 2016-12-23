from __future__ import division
import random
import file_handler
import csv
import numpy
from inverted_index import InvertedIndex
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn import svm

test_range = 10

# Loop for measuring avg mse
mse_dict = {'mlp':list(), 'tree':list(), 'svm':list()}

for i in range(1,(test_range+1)):
	print "\nRunning test loop "+str(i)+" out of "+str(test_range)+"..."

	relevant_files = file_handler.getrelevant()
	trainset = random.sample(relevant_files, 103)
	testset = [x for x in relevant_files if x not in trainset]

	index = InvertedIndex(trainset)
	index.trim()
	training_samples, training_classes = index.tf_idf()
	test_samples = index.tf_idf(testset)

	# ==================================================
	# MLP classifier
	# ==================================================

	# Training step
	#print "Starting training step..."
	classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
	classifier.fit(training_samples, training_classes)
	#print "Finished training."

	# Testing step
	#print "Starting testing step..."
	result = classifier.predict(test_samples)

	misses = 0
	for i in range(0, len(testset)):
		target_class = 0 if testset[i][1] == 'j' else 1
		predicted_class = result[i]
		if target_class != predicted_class:
			misses += 1

	mse = misses / len(testset)
	print "MLP::mse: \t\t"+str(mse)
	mse_dict['mlp'].append(mse)

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
	result = classifier.predict(test_samples)

	misses = 0
	for i in range(0, len(testset)):
		target_class = 0 if testset[i][1] == 'j' else 1
		predicted_class = result[i]
		if target_class != predicted_class:
			misses += 1

	mse = misses / len(testset)
	print "Decision tree::mse: \t"+str(mse)
	mse_dict['tree'].append(mse)

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
	result = classifier.predict(test_samples)

	misses = 0
	for i in range(0, len(testset)):
		target_class = 0 if testset[i][1] == 'j' else 1
		predicted_class = result[i]
		if target_class != predicted_class:
			misses += 1

	mse = misses / len(testset)
	print "SVM::mse: \t\t"+str(mse)
	mse_dict['svm'].append(mse)


print "\n\nAverage mse: "
print "\tMLP -> \t\t"+str(numpy.mean(mse_dict['mlp']))
print "\tDecision tree -> \t"+str(numpy.mean(mse_dict['tree']))
print "\tSVM -> \t\t"+str(numpy.mean(mse_dict['svm']))

# with open("dump.csv","wb") as f:
#	 writer = csv.writer(f)
#	 writer.writerows(matrix)