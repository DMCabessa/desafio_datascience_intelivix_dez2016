import operator
from nltk.corpus import brown

categories_count = {x: 0 for x in brown.categories()}

_id = u'cats.txt'
file = brown.open(_id)

for line in file:
	category = line.rstrip('\n').split()[1]
	categories_count[category] += 1

sortedlist = sorted(categories_count.items(), key=operator.itemgetter(1), reverse=True)
return sortedlist