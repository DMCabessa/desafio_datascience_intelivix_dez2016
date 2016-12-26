import random
import file_handler

from sklearn import svm
from inverted_index import InvertedIndex
from sklearn.grid_search import GridSearchCV

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from collections import Counter

test_range = 10
results = {'C' : [], 'kernel' : [], 'gamma' : []}

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
	# Support vector machine classifier
	# ==================================================

	# SVM parameters
	params = {
	'C' : np.array([1.0,10.0,100.0]),
	'kernel' : ["linear","poly","rbf","sigmoid"],
	'gamma' : np.array([1e-3,1e-4,1e-5])}

	for param in params.keys():
		print "========================================"
		print "Testing values for '"+param+"'"
		print "========================================"
		classifier = svm.SVC()
		grid = GridSearchCV(estimator=classifier, #verbose=10,
			param_grid={param:params[param]})
		grid.fit(training_samples, training_classes)
		#print(grid)
		print "> Best score: "+str(grid.best_score_)
		print "> Best param: "+str(getattr(grid.best_estimator_,param))
		results[param].append(str(getattr(grid.best_estimator_,param)))

print "\n\n"
for arg in results.keys():
	print "Best value for '"+arg+"': "+Counter(results[arg]).most_common()[0][0]