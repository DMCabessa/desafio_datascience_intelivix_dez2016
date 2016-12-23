from __future__ import division
import random
import file_handler
import csv
from inverted_index import InvertedIndex
from sklearn.neural_network import MLPClassifier

relevant_files = file_handler.getrelevant()
trainset = random.sample(relevant_files, 103)
testset = [x for x in relevant_files if x not in trainset]

index = InvertedIndex(trainset)
training_samples, training_classes = index.tf_idf()
test_samples = index.tf_idf(testset)

# ==================================================
# MLP classifier
# ==================================================

# Training step
print "Starting training step..."
classifier_01 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
classifier_01.fit(training_samples, training_classes)
print "Finished training."

# Testing step
print "Starting testing step..."
result = classifier_01.predict(test_samples)

hits = 0
for i in range(0, len(testset)):
	target_class = 0 if testset[i][1] == 'j' else 1
	predicted_class = result[i]
	if target_class == predicted_class:
		hits += 1

hitrate = hits / len(testset) * 100
print "========================================"
print "======= Hitrate = "+str(hitrate)+"% ======="
print "========================================"

# with open("dump.csv","wb") as f:
#	 writer = csv.writer(f)
#	 writer.writerows(matrix)