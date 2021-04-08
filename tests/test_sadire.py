import umap
import numpy as np 

from sklearn.datasets import load_iris

import sadire

def test_sadire():

	iris_data = load_iris()

	X, y = iris_data.data, iris_data.target

	reducer = umap.UMAP(random_state=0)
	emb = reducer.fit_transform(X)

	model = sadire.SADIRE(2, 5)
	reps = model.fit_transform(emb)

	assert len(reps) > 0


	model = sadire.SADIRE(1, 2)
	reps = model.fit_transform(emb)

	assert len(reps) > 0

	model = sadire.SADIRE(3, 5)
	reps = model.fit_transform(emb)

	assert len(reps) > 0