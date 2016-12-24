import random
import file_handler

from sklearn import tree
from inverted_index import InvertedIndex
from sklearn.grid_search import GridSearchCV

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

test_range = 5
results = {'criterion' : [], 'max_features' : [],
 	'min_samples_split' : [], 'min_samples_leaf' : []}

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
	# Decision tree classifier
	# ==================================================

	# Decision tree parameters
	params = {
	'criterion' : ['gini','entropy'],
	'max_features' : ['auto', 'sqrt', 'log2', None],
	'min_samples_split' : np.array([2,4,6]),
	'min_samples_leaf' : np.array([1,2,3])
	}

	for param in params.keys():
		print "========================================"
		print "Testing values for '"+param+"'"
		print "========================================"
		classifier = tree.DecisionTreeClassifier()
		grid = GridSearchCV(estimator=classifier, #verbose=10,
			param_grid={param:params[param]})
		grid.fit(training_samples, training_classes)
		#print(grid)
		print "> Best score: "+str(grid.best_score_)
		print "> Best param: "+str(getattr(grid.best_estimator_,param))
		results[param].append(str(getattr(grid.best_estimator_,param)))