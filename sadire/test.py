import umap 
import numpy as np 
import sampling as sp 

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

def normalizeVertex(coords, begin=10, end=650):
	maxX = coords[0][0]
	minX = coords[0][0]
	maxY = coords[0][1]
	minY = coords[0][1]

	for i in range(len(coords)):
		if maxX < coords[i][0]:
			maxX = coords[i][0]
		elif minX > coords[i][0]:
			minX = coords[i][0]

		if maxY < coords[i][1]:
			maxY = coords[i][1]
		elif minY > coords[i][1]:
			minY = coords[i][1]
    
	endX = ((maxX - minX) * end);
	if maxY != minY:
		endX = ((maxX - minX) * end) / (maxY - minY)


	for i in range(len(coords)):
		if maxX != minX:
			coords[i][0] = (((coords[i][0] - minX) / (maxX - minX)) * (endX - begin)) + begin + 70
		else:
			coords[i][0] = begin + 70

		if maxY != minY:
			coords[i][1] = ((((coords[i][1] - minY) / (maxY - minY)) * (end - begin)) + begin) + 70
		else:
			coords[i][1] = begin + 70

	return coords


iris_data = load_iris()

X, y = iris_data.data, iris_data.target
feature_names = np.array(iris_data.feature_names)

X = np.load('MNIST_10500.npy')
y = np.load('MNIST_10500_label.npy').astype(int)



reducer = umap.UMAP(random_state=0)
emb = reducer.fit_transform(X)

plt.scatter(emb[:, 0], emb[:, 1], c=y)
plt.show()
plt.clf()
sad = sp.SADIRE(2, 5)
representatives = sad.fit_transform(emb)
print(representatives)
plt.scatter(emb[representatives, 0], emb[representatives, 1], c=y[representatives])
plt.show()
plt.clf()
