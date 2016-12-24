import random
import file_handler

from inverted_index import InvertedIndex
from sklearn.neural_network import MLPClassifier
from sklearn.grid_search import GridSearchCV

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

test_range = 5
results = {'hidden_layer_sizes' : [], 'activation' : [], 'solver' : [],
	'alpha' : [], 'learning_rate' : [], 'learning_rate_init' :  []}

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

	# MLP parameters
	params = {
	'hidden_layer_sizes' : np.array([(50,),(100,),(200)]),
	'activation' : np.array(['identity','logistic','tanh','relu']),
	'solver' : np.array(['lbfgs','sgd','adam']),
	'alpha' : np.array([0.001,0.0001,0.00001]),
	'learning_rate' : ['constant','invscaling','adaptive'],
	'learning_rate_init' :  np.array([0.01,0.001,0.0001])}

	for param in params.keys():
		print "========================================"
		print "Testing values for '"+param+"'"
		print "========================================"
		classifier = MLPClassifier()
		grid = GridSearchCV(estimator=classifier, #verbose=10,
			param_grid={param:params[param]})
		grid.fit(training_samples, training_classes)
		#print(grid)
		print "> Best score: "+str(grid.best_score_)
		print "> Best param: "+str(getattr(grid.best_estimator_,param))
		results[param].append(str(getattr(grid.best_estimator_,param)))