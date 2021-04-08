import numpy as np
import math

from pyqtree import Index

class SADIRE(object):
	"""docstring for SADIRE"""
	

	def __init__(self, alpha, beta, type_search=1, normalize=True, begin=10, end=650):
		super(SADIRE, self).__init__()
		self.alpha = alpha
		self.size = beta 
		self.size_search = 1
		self.type_search = type_search
		self.normalize = normalize
		self.begin = begin
		self.end = end

	def check_data(self, X):
		if not isinstance(X, np.ndarray):
			raise Exception("X must be an NumPy array!")
		if not len(X.shape) == 2 or not len(X[0]) == 2:
			raise Exception("X must be an 2D array!")		

	def fit_transform(self, X):

		self.check_data(X)
		X_copy = np.copy(X)
		if self.normalize:
			X_copy = self._normalizeVertex(X_copy, self.begin, self.end)

		max_x, max_y = X_copy.max(axis=0)
		quadtree = Index(bbox=[0, 0, max_x+10, max_y+10])

		for i, x in enumerate(X_copy):
			bbox = (x[0], x[1], x[0], x[1])
			quadtree.insert(item={'point': x, 'bbox': bbox, 'id': i}, bbox=bbox)


		maxy = int(max_x/self.size+50)
		maxx = int(max_y/self.size+50)

		cells = []

		for i in range(1, maxy+1):
			for j in range(1, maxx+1):
				bbox = ((i-1)*self.size, (j-1)*self.size, i*self.size, j*self.size)
				d = quadtree.intersect(bbox)
				if len(d) != 0:
					cells.append({
						'x': (i-1),
						'y': (j-1),
						'points': d 
					})

		samples = self._sample(quadtree, cells, self.size, self.alpha, self.size_search, maxx, maxy, 1)

		return np.array([u['id'] for u in samples])

	def _sample(self, quadtree, cells, delta, alpha, size_search, maxx, maxy, fraction):

		sampled_set = []
		A = []

		for i in range(len(cells)):
			bbox = (cells[i]['x']*delta, cells[i]['y']*delta, (cells[i]['x']+1)*delta, (cells[i]['y']+1)*delta)
			points = quadtree.intersect(bbox)
			
			if size_search == 1:
				A.append({
					'box': cells[i],
					'representative': self._get_point(points),
					'points': points,
					'active': True
				})
			# TODO implement for percentage


		A.sort(key=lambda x: len(x['points']), reverse=True)

		for i in range(len(A)):

			if A[i]['active']:

				has_neighbors = False
				d_matrix = 2*alpha+1
				mid_point = (d_matrix-1)/2

				for j in range(d_matrix):
					for k in range(d_matrix):
						dx = (A[i]['box']['x']*delta - mid_point*delta) + j*delta
						dy = (A[i]['box']['y']*delta - mid_point*delta) + k*delta

						if dx == A[i]['box']['x']*delta and dy == A[i]['box']['y']*delta:
							continue

						if dx >= 0 and dx <= math.ceil(maxy*delta) and dy >= 0 and dy <= math.ceil(maxx*delta):
							neighbor = self._findbox(A, dx, dy, delta)
							if neighbor != None:
								neighbor['active'] = False 
								has_neighbors = True

				if has_neighbors:
					A[i]['active'] = False 
					sampled_set.append(A[i]['representative'])

		for i in range(len(A)):
			if A[i]['active']:
				A[i]['active'] = False 

				point = self._get_point(A[i]['points'])

				min_distance = 1000000.0
				neighbor = None 

				for j in range(len(sampled_set)):

					d = abs(sampled_set[j]['point'][0]-point['point'][0]) + abs(sampled_set[j]['point'][1]-point['point'][1])

					if d <= (2.0/fraction)*delta and min_distance > d:
						min_distance = d
						neighbor = sampled_set[j]

				if neighbor == None:
					if size_search == 1:
						sampled_set.append(point)
					# TODO implement for percentage


		return sampled_set

	def _findbox(self, A, x, y, delta):
		for i in range(len(A)):
			if A[i]['active'] and A[i]['box']['x']*delta == x and A[i]['box']['y']*delta == y:
				return A[i]

		return None

	def _get_point(self, points):
		if self.type_search == 1:
			
			return points[0]

		elif self.type_search == 2:

			x = 0.0 
			y = 0.0

			for p in points:
				x += p['point'][0]
				y += p['point'][1]

			x /= len(points)
			y /= len(points)

			medoid = points[0]
			distance = math.sqrt(math.pow(x-medoid['point'][0], 2.0) + math.pow(y-medoid['point'][1], 2.0))
			
			for j in len(points):
				d = math.sqrt(math.pow(x-points[j]['point'][0], 2.0) + math.pow(y-points[j]['point'][1], 2.0))
				if d < distance:
					distance = d 
					medoid = points[j]

			return medoid

		else:

			return points[random.randint(0, len(points)-1)] 


	def _normalizeVertex(self, coords, begin=10, end=650):
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


		
















