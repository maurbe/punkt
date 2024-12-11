from __future__ import print_function
import numpy as np
import tqdm
cimport numpy as np
cimport cython

#def initialize():
#	np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
def _account_for_pbc(int a, int gridnum, int periodic):
	cdef int an = a
	cdef float fraction = -1.0
		
	if (a < 0):
		an = a + gridnum   # e.g. 127 = -1 + 128
		if periodic == 0: # if not periodic, dont deposit into this (otherwise wrapped) cell
			fraction = 0.0

	elif (a >= gridnum):
		an = a - gridnum  # e.g. 0 = 128 - 128
		if periodic == 0:
			fraction = 0.0
		
	return an, fraction


@cython.boundscheck(False)
@cython.wraparound(False)
def _divide_vector_nd_by_scalar(np.ndarray[np.float32_t, ndim=1] vec, np.float32_t scalar):

	cdef int rows = vec.shape[0]
	cdef int i
	
	for i in range(rows):
		vec[i] /= scalar
	
	return vec


"""
@cython.boundscheck(False)
@cython.wraparound(False)
def _divide_matrix_2d_by_scalar(np.ndarray[np.float32_t, ndim=2] mat, np.float32_t scalar):

	cdef int rows = mat.shape[0]
	cdef int cols = mat.shape[1]
	cdef int i, j
	
	for i in range(rows):
		for j in range(cols):
			mat[i, j] /= scalar
	
	return mat


@cython.boundscheck(False)
@cython.wraparound(False)
def _multiply_matrix_2d_by_scalar(np.ndarray[np.float32_t, ndim=2] mat, np.float32_t scalar):

	cdef int rows = mat.shape[0]
	cdef int cols = mat.shape[1]
	cdef int i, j
	
	for i in range(rows):
		for j in range(cols):
			mat[i, j] *= scalar
	
	return mat


@cython.boundscheck(False)
@cython.wraparound(False)
def _matrix_vector_multiply(np.ndarray[np.float32_t, ndim=2] matrix, np.ndarray[np.float32_t, ndim=1] vector):

	cdef np.ndarray[np.float32_t, ndim=1, mode='c'] result = np.zeros(2, dtype=np.float32)
	result[0] = matrix[0, 0] * vector[0] + matrix[0, 1] * vector[1]
	result[1] = matrix[1, 0] * vector[0] + matrix[1, 1] * vector[1]

	return result


@cython.boundscheck(False)
@cython.wraparound(False)
def _vector_norm(np.ndarray[np.float32_t, ndim=1] arr):

	cdef np.float32_t norm_sq = 0.0
	cdef int size = arr.shape[0]
	cdef int i
	
	for i in range(size):
		norm_sq += arr[i] * arr[i]
	
	return norm_sq ** 0.5
"""



@cython.boundscheck(False)
@cython.wraparound(False)
def _extract_max_value_1d(np.ndarray[np.float32_t, ndim=1] arr):

	cdef int size = arr.shape[0]
	cdef int i
	cdef np.float32_t max_val = arr[0]
	
	for i in range(size):
		if arr[i] > max_val:
			max_val = arr[i]
	
	return max_val


@cython.boundscheck(False)
@cython.wraparound(False)
def anisotropic_kernel_deposition_2d(
		  np.ndarray[np.float32_t, ndim=2] pos,
		  np.ndarray[np.float32_t, ndim=3] hmat_eigvecs,
		  np.ndarray[np.float32_t, ndim=2] hmat_eigvals,
		  np.ndarray[np.float32_t, ndim=2] quantities,
		  np.ndarray[np.float32_t, ndim=1] extent,
		  np.int32_t gridnum,
		  np.int32_t periodic
		  ):
	
	cdef int n, i, j, a, b, an, bn, f
	cdef int num_left, num_bottom, num_right, num_top, num_fields
	cdef np.float32_t xpos, ypos
	cdef np.float32_t krs, cellSize, boxsize, fraction, F
	cdef np.float32_t q, detH, xi_1, xi_2
	cdef np.ndarray[np.float32_t, ndim=1, mode='c'] r  = np.zeros(2, dtype=np.float32)

	num_fields = quantities.shape[1]
	cdef np.ndarray[np.float32_t, ndim=3, mode='c'] fields  = np.zeros((gridnum, gridnum, num_fields), dtype=np.float32)
	cdef np.ndarray[np.float32_t, ndim=2, mode='c'] weights = np.zeros((gridnum, gridnum), dtype=np.float32)
	cellSize = (extent[1] - extent[0]) / (<np.float32_t> gridnum)
	boxsize  =  extent[1] - extent[0]


	for n in tqdm.trange(len(pos)):

		# normalize length scales
		# the eigvals need to be scaled but NOT the normalized unit vectors!!
		# they are the same in every frame!
		hmat_eigvecs_norm = hmat_eigvecs[n]
		hmat_eigvals_norm = _divide_vector_nd_by_scalar(hmat_eigvals[n], cellSize)
		krs				  = _extract_max_value_1d(hmat_eigvals_norm) * 3 # 2 kernel has compact support until 3h!

		xpos = pos[n, 0] / cellSize
		ypos = pos[n, 1] / cellSize
		
		# mother cell, in which the particle resides
		i = int(xpos)
		j = int(ypos)
		
		# now we have to identify all the neighbors cells in all four directions 
		# that are withing the reach of the normalized kernel size krs
		num_left   = i - int(xpos - krs)
		num_bottom = j - int(ypos - krs)
		num_right  = int(xpos + krs) - i
		num_top    = int(ypos + krs) - j

		for a in range(i - num_left, i + num_right + 1):
			for b in range(j - num_bottom, j + num_top + 1):

				# here we can check whether the kernel is completely inside the mother cell
				# if, yes the fraction is just 1.0
				if (num_left == 0) & (num_bottom == 0) & (num_right == 0) & (num_top == 0):
					fraction = 1.0
					an = a
					bn = b

				else:
					# approximate the integral with the area of the cell * kernel(midpoint)
					# account for pbc
					an, fraction = _account_for_pbc(a, gridnum, periodic)
					bn, fraction = _account_for_pbc(b, gridnum, periodic)

					# distance to midpoint of cell
					dist_x = (xpos - (<np.float32_t> a + 0.5))
					dist_y = (ypos - (<np.float32_t> b + 0.5))

					if periodic:
						if dist_x > boxsize / 2.0:
							dist_x -= boxsize
						if dist_y > boxsize / 2.0:
							dist_y -= boxsize

					xi_1 = (hmat_eigvecs_norm[0, 0] * dist_x + hmat_eigvecs_norm[0, 1] * dist_y) / hmat_eigvals_norm[0]
					xi_2 = (hmat_eigvecs_norm[1, 0] * dist_x + hmat_eigvecs_norm[1, 1] * dist_y) / hmat_eigvals_norm[1]
					q = (xi_1 ** 2 + xi_2 ** 2) ** 0.5

					detH = hmat_eigvals_norm[0] * hmat_eigvals_norm[1]
					fraction = _quintic_spline(q) * 7.0 / (478 * 3.141592653589793 * detH) # * 1.0 * 1.0
				
				# deposit multiple fields
				for f in range(num_fields):
					fields[an, bn, f] += fraction * quantities[n, f]
				weights[an, bn] += fraction

	return fields, weights


@cython.boundscheck(False)
@cython.wraparound(False)
def anisotropic_kernel_deposition_3d(
		  np.ndarray[np.float32_t, ndim=2] pos,
		  np.ndarray[np.float32_t, ndim=3] hmat_eigvecs,
		  np.ndarray[np.float32_t, ndim=2] hmat_eigvals,
		  np.ndarray[np.float32_t, ndim=2] quantities,
		  np.ndarray[np.float32_t, ndim=1] extent,
		  np.int32_t gridnum,
		  np.int32_t periodic
		  ):
	
	cdef int n, i, j, k, a, b, c, an, bn, cn, f
	cdef int num_left, num_bottom, num_right, num_top, num_front, num_back, num_fields
	cdef np.float32_t xpos, ypos, zpos
	cdef np.float32_t krs, cellSize, boxsize, fraction
	cdef np.float32_t q, detH, xi_1, xi_2, xi_3
	cdef np.ndarray[np.float32_t, ndim=1, mode='c'] r  = np.zeros(2, dtype=np.float32)

	num_fields = quantities.shape[1]
	cdef np.ndarray[np.float32_t, ndim=4, mode='c'] fields  = np.zeros((gridnum, gridnum, gridnum, num_fields), dtype=np.float32)
	cdef np.ndarray[np.float32_t, ndim=3, mode='c'] weights = np.zeros((gridnum, gridnum, gridnum), dtype=np.float32)
	cellSize = (extent[1] - extent[0]) / (<np.float32_t> gridnum)
	boxsize  =  extent[1] - extent[0]


	for n in tqdm.trange(len(pos)):

		# normalize length scales
		# the eigvals need to be scaled but NOT the normalized unit vectors!!
		# they are the same in every frame!
		hmat_eigvecs_norm = hmat_eigvecs[n]
		hmat_eigvals_norm = _divide_vector_nd_by_scalar(hmat_eigvals[n], cellSize)
		krs				  = _extract_max_value_1d(hmat_eigvals_norm) * 3 # 2 kernel has compact support until 3h!

		xpos = pos[n, 0] / cellSize
		ypos = pos[n, 1] / cellSize
		zpos = pos[n, 2] / cellSize
		
		# mother cell, in which the particle resides
		i = int(xpos)
		j = int(ypos)
		k = int(zpos)
		
		# now we have to identify all the neighbors cells in all four directions 
		# that are withing the reach of the normalized kernel size krs
		num_left   = i - int(xpos - krs)
		num_bottom = j - int(ypos - krs)
		num_front  = k - int(zpos - krs)
		num_right  = int(xpos + krs) - i
		num_top    = int(ypos + krs) - j
		num_back   = int(zpos + krs) - k

		for a in range(i - num_left, i + num_right + 1):
			for b in range(j - num_bottom, j + num_top + 1):
				for c in range(k - num_front, k + num_back + 1):

					# here we can check whether the kernel is completely inside the mother cell
					# if, yes the fraction is just 1.0
					if (num_left == 0) & (num_bottom == 0) & (num_right == 0) & (num_top == 0) & (num_front == 0) & (num_back == 0):
						fraction = 1.0
						an = a
						bn = b
						cn = c

					else:
						# approximate the integral with the area of the cell * kernel(midpoint)
						# account for pbc
						an, fraction = _account_for_pbc(a, gridnum, periodic)
						bn, fraction = _account_for_pbc(b, gridnum, periodic)
						cn, fraction = _account_for_pbc(c, gridnum, periodic)

						# distance to midpoint of cell
						dist_x = (xpos - (<np.float32_t> a + 0.5))
						dist_y = (ypos - (<np.float32_t> b + 0.5))
						dist_z = (zpos - (<np.float32_t> c + 0.5))

						if periodic:
							if dist_x > boxsize / 2.0:
								dist_x -= boxsize
							if dist_y > boxsize / 2.0:
								dist_y -= boxsize
							if dist_z > boxsize / 2.0:
								dist_z -= boxsize

						xi_1 = (hmat_eigvecs_norm[0, 0] * dist_x + hmat_eigvecs_norm[0, 1] * dist_y + hmat_eigvecs_norm[0, 2] * dist_z) / hmat_eigvals_norm[0]
						xi_2 = (hmat_eigvecs_norm[1, 0] * dist_x + hmat_eigvecs_norm[1, 1] * dist_y + hmat_eigvecs_norm[1, 2] * dist_z) / hmat_eigvals_norm[1]
						xi_3 = (hmat_eigvecs_norm[2, 0] * dist_x + hmat_eigvecs_norm[2, 1] * dist_y + hmat_eigvecs_norm[2, 2] * dist_z) / hmat_eigvals_norm[2]
						q = (xi_1 ** 2 + xi_2 ** 2 + xi_3 ** 2) ** 0.5

						detH = hmat_eigvals_norm[0] * hmat_eigvals_norm[1] * hmat_eigvals_norm[2]
						fraction = _quintic_spline(q) / (120 * 3.141592653589793 * detH) # * 1.0 * 1.0 * 1.0
					
					# deposit multiple fields
					for f in range(num_fields):
						fields[an, bn, cn, f] += fraction * quantities[n, f]
					weights[an, bn, cn] += fraction

	return fields, weights


@cython.boundscheck(False)
@cython.wraparound(False)
def _sigma(int dim, float h):
	cdef float sigma_value = 0.0
	
	if dim == 1:
		sigma_value = 1.0 / (120 * h)
	elif dim == 2:
		sigma_value = 7.0 / (478 * 3.141592653589793 * h ** 2)
	elif dim == 3:
		sigma_value = 1.0 / (120 * 3.141592653589793 * h ** 3)
	
	return sigma_value


@cython.boundscheck(False)
@cython.wraparound(False)
def _quintic_spline(np.float32_t q):

	if 0.0 <= q <= 1.0:
		return (3 - q)**5 - 6*(2 - q)**5 + 15*(1 - q)**5
	elif 1.0 < q <= 2.0:
		return (3 - q)**5 - 6*(2 - q)**5
	elif 2.0 < q <= 3.0:
		return (3 - q)**5
	else:
		return 0.0


@cython.boundscheck(False)
@cython.wraparound(False)
def isotropic_kernel_deposition_2d(
		  np.ndarray[np.float32_t, ndim=2] pos,
		  np.ndarray[np.float32_t, ndim=1] hsm,
		  np.ndarray[np.float32_t, ndim=2] quantities,
		  np.ndarray[np.float32_t, ndim=1] extent,
		  np.int32_t gridnum,
		  np.int32_t periodic
		  ):
	
	cdef int n, i, j, a, b, an, bn, f
	cdef int num_left, num_bottom, num_right, num_top, num_fields
	cdef np.float32_t xpos, ypos
	cdef np.float32_t krs, cellSize, boxsize, fraction
	cdef np.float32_t q, r

	num_fields = quantities.shape[1]
	cdef np.ndarray[np.float32_t, ndim=3, mode='c'] fields  = np.zeros((gridnum, gridnum, num_fields), dtype=np.float32)
	cdef np.ndarray[np.float32_t, ndim=2, mode='c'] weights = np.zeros((gridnum, gridnum), dtype=np.float32)
	cellSize = (extent[1] - extent[0]) / (<np.float32_t> gridnum)
	boxsize  =  extent[1] - extent[0]

	for n in tqdm.trange(len(pos)):

		# normalize length scales
		hsn  = hsm[n] / cellSize
		krs  = hsn * 3 # 2 kernel has compact support until 3h!

		xpos = pos[n, 0] / cellSize
		ypos = pos[n, 1] / cellSize
		
		# mother cell, in which the particle resides
		i = int(xpos)
		j = int(ypos)
		
		# now we have to identify all the neighbors cells in all four directions 
		# that are withing the reach of the normalized kernel size krs
		num_left   = i - int(xpos - krs)
		num_bottom = j - int(ypos - krs)
		num_right  = int(xpos + krs) - i
		num_top    = int(ypos + krs) - j

		
		for a in range(i - num_left, i + num_right + 1):
			for b in range(j - num_bottom, j + num_top + 1):

				# here we can check whether the kernel is completely inside the mother cell
				# if, yes the fraction is just 1.0
				if (num_left == 0) & (num_bottom == 0) & (num_right == 0) & (num_top == 0):
					fraction = 1.0
					an = a
					bn = b

				else:
					# approximate the integral with the area of the cell * kernel(midpoint)
					# account for pbc
					an, fraction = _account_for_pbc(a, gridnum, periodic)
					bn, fraction = _account_for_pbc(b, gridnum, periodic)

					# distance to midpoint of cell
					dist_x = (xpos - (<np.float32_t> a + 0.5))
					dist_y = (ypos - (<np.float32_t> b + 0.5))

					if periodic:
						if dist_x > boxsize / 2.0:
							dist_x -= boxsize
						if dist_y > boxsize / 2.0:
							dist_y -= boxsize

					r = (dist_x ** 2 + dist_y ** 2) ** 0.5
					q = r / hsn

					# kernel * area (= cellSize^2 = 1^2) = volume = fraction
					fraction = _quintic_spline(q) * _sigma(dim=2, h=hsn) # * 1.0 * 1.0
				
				# deposit multiple fields
				for f in range(num_fields):
					fields[an, bn, f] += fraction * quantities[n, f]
				weights[an, bn] += fraction
	
	return fields, weights


@cython.boundscheck(False)
@cython.wraparound(False)
def isotropic_kernel_deposition_3d(
		  np.ndarray[np.float32_t, ndim=2] pos,
		  np.ndarray[np.float32_t, ndim=1] hsm,
		  np.ndarray[np.float32_t, ndim=2] quantities,
		  np.ndarray[np.float32_t, ndim=1] extent,
		  np.int32_t gridnum,
		  np.int32_t periodic
		  ):
	
	cdef int n, i, j, k, a, b, c, an, bn, cn, f
	cdef int num_left, num_bottom, num_right, num_top, num_front, num_back, num_fields
	cdef np.float32_t xpos, ypos, zpos
	cdef np.float32_t krs, cellSize, boxsize, fraction
	cdef np.float32_t q, r

	num_fields = quantities.shape[1]
	cdef np.ndarray[np.float32_t, ndim=4, mode='c'] fields  = np.zeros((gridnum, gridnum, gridnum, num_fields), dtype=np.float32)
	cdef np.ndarray[np.float32_t, ndim=3, mode='c'] weights = np.zeros((gridnum, gridnum, gridnum), dtype=np.float32)
	cellSize = (extent[1] - extent[0]) / (<np.float32_t> gridnum)
	boxsize  =  extent[1] - extent[0]

	for n in tqdm.trange(len(pos)):

		# normalize length scales
		hsn  = hsm[n] / cellSize
		krs  = hsn * 3 # 2 kernel has compact support until 3h!

		xpos = pos[n, 0] / cellSize
		ypos = pos[n, 1] / cellSize
		zpos = pos[n, 2] / cellSize
		
		# mother cell, in which the particle resides
		i = int(xpos)
		j = int(ypos)
		k = int(zpos)
		
		# now we have to identify all the neighbors cells in all four directions 
		# that are withing the reach of the normalized kernel size krs
		num_left   = i - int(xpos - krs)
		num_bottom = j - int(ypos - krs)
		num_front  = k - int(zpos - krs)
		num_right  = int(xpos + krs) - i
		num_top    = int(ypos + krs) - j
		num_back   = int(zpos + krs) - k

		
		for a in range(i - num_left, i + num_right + 1):
			for b in range(j - num_bottom, j + num_top + 1):
				for c in range(k - num_front, k + num_back + 1):

					# here we can check whether the kernel is completely inside the mother cell
					# if, yes the fraction is just 1.0
					if (num_left == 0) & (num_bottom == 0) & (num_right == 0) & (num_top == 0) & (num_front == 0) & (num_back == 0):
						fraction = 1.0
						an = a
						bn = b
						cn = c

					else:
						# approximate the integral with the area of the cell * kernel(midpoint)
						# account for pbc
						an, fraction = _account_for_pbc(a, gridnum, periodic)
						bn, fraction = _account_for_pbc(b, gridnum, periodic)
						cn, fraction = _account_for_pbc(c, gridnum, periodic)

						# distance to midpoint of cell
						dist_x = (xpos - (<np.float32_t> a + 0.5))
						dist_y = (ypos - (<np.float32_t> b + 0.5))
						dist_z = (zpos - (<np.float32_t> c + 0.5))

						if periodic:
							if dist_x > boxsize / 2.0:
								dist_x -= boxsize
							if dist_y > boxsize / 2.0:
								dist_y -= boxsize
							if dist_z > boxsize / 2.0:
								dist_z -= boxsize

						r = (dist_x ** 2 + dist_y ** 2 + dist_z ** 2) ** 0.5
						q = r / hsn

						# kernel * area (= cellSize^3 = 1^3) = volume = fraction
						fraction = _quintic_spline(q) * _sigma(dim=3, h=hsn) # * 1.0 * 1.0 * 1.0
					
					# deposit multiple fields
					for f in range(num_fields):
						fields[an, bn, cn, f] += fraction * quantities[n, f]
					weights[an, bn, cn] += fraction
	
	return fields, weights


@cython.boundscheck(False)
@cython.wraparound(False)
def cic_2d(np.ndarray[np.float32_t, ndim=2] pos,
		   np.ndarray[np.float32_t, ndim=2] quantities,
		   np.ndarray[np.float32_t, ndim=1] extent,
		   np.int32_t gridnum,
		   np.int32_t periodic
		   ):

	cdef int i, j, i_, j_, n, f, num_fields
	cdef np.float32_t xpos, ypos 
	cdef np.float32_t cellSize
	cdef np.float32_t dx, dy, dx_, dy_

	num_fields = quantities.shape[1]
	cdef np.ndarray[np.float32_t, ndim=3, mode='c'] fields  = np.zeros((gridnum, gridnum, num_fields), dtype=np.float32)
	cdef np.ndarray[np.float32_t, ndim=2, mode='c'] weights = np.zeros((gridnum, gridnum), dtype=np.float32)
	cellSize = (extent[1] - extent[0]) / (<np.float32_t> gridnum)


	for n in tqdm.trange(len(pos)):
		# Compute the position of the central cell and round to avoid floating point issues
		# This transforms the coordinates into grid dimensions (e.g. 15000 -> 512)
		xpos = round((pos[n, 0] - extent[0]) / cellSize, 5)   # Example: posx = -0.49, 0.49 -> xpos = 0.2, 19.8
		ypos = round((pos[n, 1] - extent[0]) / cellSize, 5)


		# debugging: only deposit particles that are wrapped inside the deposition domain
		if ((xpos < 0.0) or (xpos > (<np.float32_t> gridnum))) or \
		   ((ypos < 0.0) or (ypos > (<np.float32_t> gridnum))):
			continue

		
		i  = int(xpos + 0.4999)
		j  = int(ypos + 0.4999)
		i_ = i-1
		j_ = j-1


		# Compute the weights
		dx = (xpos + 0.5) - (<np.float32_t> i)
		dy = (ypos + 0.5) - (<np.float32_t> j)
		dx_=  1.0 - dx
		dy_=  1.0 - dy

		# accounting for pbc
		if periodic == 1:
			if (i == 0):
				i_ = gridnum-1
			if (i == gridnum):
				i = 0

			if (j == 0):
				j_ = gridnum-1
			if (j == gridnum):
				j = 0

		# if not periodic, perform the boundary check and only deposit into the cells inside domain
		if periodic == 0:
			if (xpos < 0.5):
				continue # dx_ = 0.0
			if (xpos > (<np.float32_t> gridnum) - 0.5):
				continue # dx = 0.0
			if (ypos < 0.5):
				continue # dy_ = 0.0
			if (ypos > (<np.float32_t> gridnum) - 0.5):
				continue # dy = 0.0
				

		# Fill the fields
		for f in range(num_fields):
			fields[i_, j_, f] += quantities[n, f] * dx_ * dy_
			fields[i , j_, f] += quantities[n, f] * dx  * dy_
			fields[i_, j , f] += quantities[n, f] * dx_ * dy
			fields[i , j , f] += quantities[n, f] * dx  * dy

		# Keep track of the weights
		weights[i_, j_] += dx_ * dy_
		weights[i , j_] += dx  * dy_
		weights[i_, j ] += dx_ * dy
		weights[i , j ] += dx  * dy

	return fields, weights
	

@cython.boundscheck(False)
@cython.wraparound(False)
def cic_3d(np.ndarray[np.float32_t, ndim=2] pos,
		   np.ndarray[np.float32_t, ndim=2] quantities,
		   np.ndarray[np.float32_t, ndim=1] extent,
		   np.int32_t gridnum,
		   np.int32_t periodic
		   ):

	cdef int i, j, k, i_, j_, k_, n, f, num_fields
	cdef np.float32_t xpos, ypos, zpos 
	cdef np.float32_t cellSize
	cdef np.float32_t dx, dy, dz, dx_, dy_, dz_

	num_fields = quantities.shape[1]
	cdef np.ndarray[np.float32_t, ndim=4, mode='c'] fields  = np.zeros((gridnum, gridnum, gridnum, num_fields), dtype=np.float32)
	cdef np.ndarray[np.float32_t, ndim=3, mode='c'] weights = np.zeros((gridnum, gridnum, gridnum), dtype=np.float32)
	cellSize = (extent[1] - extent[0]) / (<np.float32_t> gridnum)


	for n in tqdm.trange(len(pos)):
		# Compute the position of the central cell and round to avoid floating point issues
		# This transforms the coordinates into grid dimensions (e.g. 15000 -> 512)
		xpos = round((pos[n, 0] - extent[0]) / cellSize, 5)   # Example: posx = -0.49, 0.49 -> xpos = 0.2, 19.8
		ypos = round((pos[n, 1] - extent[0]) / cellSize, 5)
		zpos = round((pos[n, 2] - extent[0]) / cellSize, 5)


		# debugging: only deposit particles that are wrapped inside the deposition domain
		if ((xpos < 0.0) or (xpos > (<np.float32_t> gridnum))) or \
		   ((ypos < 0.0) or (ypos > (<np.float32_t> gridnum))) or \
		   ((zpos < 0.0) or (zpos > (<np.float32_t> gridnum))):
			continue

		
		i  = int(xpos + 0.4999)
		j  = int(ypos + 0.4999)
		k  = int(zpos + 0.4999)
		i_ = i-1
		j_ = j-1
		k_ = k-1


		# Compute the weights
		dx = (xpos + 0.5) - (<np.float32_t> i)
		dy = (ypos + 0.5) - (<np.float32_t> j)
		dz = (zpos + 0.5) - (<np.float32_t> k)
		dx_=  1.0 - dx
		dy_=  1.0 - dy
		dz_=  1.0 - dz

		# accounting for pbc
		if periodic == 1:
			if (i == 0):
				i_ = gridnum-1
			if (i == gridnum):
				i = 0

			if (j == 0):
				j_ = gridnum-1
			if (j == gridnum):
				j = 0

			if (k == 0):
				k_ = gridnum-1
			if (k == gridnum):
				k = 0


		# if not periodic, perform the boundary check and skip particles right at the boundary
		if periodic == 0:
			if (xpos < 0.5):
				continue # dx_ = 0.0
			if (xpos > (<np.float32_t> gridnum) - 0.5):
				continue # dx = 0.0
			if (ypos < 0.5):
				continue # dy_ = 0.0
			if (ypos > (<np.float32_t> gridnum) - 0.5):
				continue # dy = 0.0
			if (zpos < 0.5):
				continue # dz_ = 0.0
			if (zpos > (<np.float32_t> gridnum) - 0.5):
				continue # dz = 0.0

		# Fill the fields
		for f in range(num_fields):
			fields[i_, j_, k_, f] += quantities[n, f] * dx_ * dy_ * dz_
			fields[i , j_, k_, f] += quantities[n, f] * dx  * dy_ * dz_
			fields[i_, j  ,k_, f] += quantities[n, f] * dx_ * dy  * dz_
			fields[i , j  ,k_, f] += quantities[n, f] * dx  * dy  * dz_
			fields[i_, j_ ,k , f] += quantities[n, f] * dx_ * dy_ * dz
			fields[i , j_, k , f] += quantities[n, f] * dx  * dy_ * dz
			fields[i_, j , k , f] += quantities[n, f] * dx_ * dy  * dz
			fields[i , j , k , f] += quantities[n, f] * dx  * dy  * dz

		# Keep track of the weights
		weights[i_, j_, k_] += dx_ * dy_ * dz_
		weights[i , j_, k_] += dx  * dy_ * dz_
		weights[i_, j  ,k_] += dx_ * dy  * dz_
		weights[i , j  ,k_] += dx  * dy  * dz_
		weights[i_, j_ ,k ] += dx_ * dy_ * dz
		weights[i , j_, k ] += dx  * dy_ * dz
		weights[i_, j , k ] += dx_ * dy  * dz
		weights[i , j , k ] += dx  * dy  * dz

	return fields, weights



cdef inline np.float64_t fmin(np.float64_t f0, np.float64_t f1) nogil:
	if f0 < f1: return f0
	return f1
cdef inline np.float64_t fmax(np.float64_t f0, np.float64_t f1) nogil:
	if f0 > f1: return f0
	return f1


@cython.boundscheck(False)
@cython.wraparound(False)
def cic_2d_adaptive(np.ndarray[np.float32_t, ndim=2] pos,
					np.ndarray[np.float32_t, ndim=2] quantities,
					np.ndarray[np.float32_t, ndim=1] pcellsizesHalf,
					np.ndarray[np.float32_t, ndim=1] extent,
					np.int32_t gridnum,
					np.int32_t periodic
					):

	cdef int n, a, b, an, bn, f, num_fields
	cdef np.float32_t xpos, ypos, A, pcs, \
		c1, c2, c3, c4, e1, e2, e3, e4, \
		intersec_x, intersec_y, fraction
	
	num_fields = quantities.shape[1]
	cdef np.ndarray[np.float32_t, ndim=3, mode='c'] fields  = np.zeros((gridnum, gridnum, num_fields), dtype=np.float32)
	cdef np.ndarray[np.float32_t, ndim=2, mode='c'] weights = np.zeros((gridnum, gridnum), dtype=np.float32)
	cdef np.float32_t cellSize = (extent[1] - extent[0]) / (<np.float32_t> gridnum)


	for n in tqdm.trange(len(pos)):
		
		pcs = pcellsizesHalf[n] / cellSize # normalized
		A = (2 * pcs * 2 * pcs)

		# first we need to compute over how many cells the particle cell spans in x and y:
		# Compute the position of the central cell and round to avoid floating point issues
		xpos = round((pos[n, 0] - extent[0]) / cellSize, 5)   # Example: posx = -0.49, 0.49 -> xpos = 0.2, 19.8
		ypos = round((pos[n, 1] - extent[0]) / cellSize, 5)

		# this is the mother cell, in which the particle resides
		i  = int(xpos) 	# this is correct
		j  = int(ypos)

		# now we have to identify all the neighbors cells in all four directions that are withing
		# the reach of the tophat kernel, round(x-0.5) rounds down even for negatives!
		num_left  = i - round(xpos - pcs - 0.5)
		num_bottom= j - round(ypos - pcs - 0.5)
		num_right = int(xpos + pcs) - i
		num_top   = int(ypos + pcs) - j

		c1 = xpos - pcs
		c2 = xpos + pcs
		c3 = ypos - pcs
		c4 = ypos + pcs

		for a in range(i - num_left, i + num_right + 1): 
			for b in range(j - num_bottom, j + num_top + 1):
				
				e1 = a * 1.0
				e2 = a + 1.0
				e3 = b * 1.0
				e4 = b + 1.0

				# now compute the overlap between the cells (c1, c2, c3, c4) and (e1, e2, e3, e4)
				intersec_x = fmin(e2, c2) - fmax(e1, c1)
				intersec_y = fmin(e4, c4) - fmax(e3, c3)
				fraction = (intersec_x * intersec_y) / A

				# cannot modify a, b inplace (will change loop); have to define new vars
				an = a
				bn = b

				# account for pbc or not
				if (a < 0):
					an = a + gridnum   # e.g. 127 = -1 + 128
					if periodic == 0: # if not periodic, dont deposit into this (otherwise wrapped) cell
						fraction = 0.0

				elif (a >= gridnum):
					an = a - gridnum  # e.g. 0 = 128 - 128
					if periodic == 0:
						fraction = 0.0

				if (b < 0):
					bn = b + gridnum
					if periodic == 0:
						fraction = 0.0

				elif (b >= gridnum):
					bn = b - gridnum
					if periodic == 0:
						fraction = 0.0

				for f in range(num_fields):
					fields[an, bn, f] += quantities[n, f] * fraction
				weights[an, bn] += fraction
	
	return fields, weights


@cython.boundscheck(False)
@cython.wraparound(False)
def cic_3d_adaptive(np.ndarray[np.float32_t, ndim=2] pos,
					np.ndarray[np.float32_t, ndim=2] quantities,
					np.ndarray[np.float32_t, ndim=1] pcellsizesHalf,
					np.ndarray[np.float32_t, ndim=1] extent,
					np.int32_t gridnum,
					np.int32_t periodic
					):

	cdef int n, a, b, c, an, bn, cn, f, num_fields
	cdef np.float32_t xpos, ypos, zpos, V, pcs, \
		c1, c2, c3, c4, c5, c6, \
		e1, e2, e3, e4, e5, e6, \
		intersec_x, intersec_y, intersec_z, fraction
	cdef np.float32_t cellSize = (extent[1] - extent[0]) / (<np.float32_t> gridnum)
	
	num_fields = quantities.shape[1]
	cdef np.ndarray[np.float32_t, ndim=4, mode='c'] fields  = np.zeros((gridnum, gridnum, gridnum, num_fields), dtype=np.float32)
	cdef np.ndarray[np.float32_t, ndim=3, mode='c'] weights = np.zeros((gridnum, gridnum, gridnum), dtype=np.float32)


	for n in tqdm.trange(len(pos)):
		
		pcs = pcellsizesHalf[n] / cellSize # normalized
		V = (2 * pcs * 2 * pcs * 2 * pcs)

		# first we need to compute over how many cells the particle cell spans in x and y:
		# Compute the position of the central cell and round to avoid floating point issues
		xpos = round((pos[n, 0] - extent[0]) / cellSize, 5)   # Example: posx = -0.49, 0.49 -> xpos = 0.2, 19.8
		ypos = round((pos[n, 1] - extent[0]) / cellSize, 5)
		zpos = round((pos[n, 2] - extent[0]) / cellSize, 5)

		# this is the mother cell, in which the particle resides
		i  = int(xpos) 	# this is correct
		j  = int(ypos)
		k  = int(zpos)

		# now we have to identify all the neighbors cells in all four directions that are withing
		# the reach of the tophat kernel, round(x-0.5) rounds down even for negatives!
		num_left  = i - round(xpos - pcs - 0.5)
		num_bottom= j - round(ypos - pcs - 0.5)
		num_back  = k - round(zpos - pcs - 0.5)
		num_right = int(xpos + pcs) - i
		num_top   = int(ypos + pcs) - j
		num_fwd   = int(zpos + pcs) - k

		c1 = xpos - pcs
		c2 = xpos + pcs
		c3 = ypos - pcs
		c4 = ypos + pcs
		c5 = zpos - pcs
		c6 = zpos + pcs

		for a in range(i - num_left, i + num_right + 1): 
			for b in range(j - num_bottom, j + num_top + 1):
				for c in range(k - num_back, k + num_fwd + 1):
				
					e1 = a * 1.0
					e2 = a + 1.0
					e3 = b * 1.0
					e4 = b + 1.0
					e5 = c * 1.0
					e6 = c + 1.0

					# now compute the overlap between the cells (c1, c2, c3, c4) and (e1, e2, e3, e4)
					intersec_x = fmin(e2, c2) - fmax(e1, c1)
					intersec_y = fmin(e4, c4) - fmax(e3, c3)
					intersec_z = fmin(e6, c6) - fmax(e5, c5)
					fraction = (intersec_x * intersec_y * intersec_z) / V

					# cannot modify a, b inplace (will change loop); have to define new vars
					an = a
					bn = b
					cn = c

					# account for pbc or not
					if (a < 0):
						an = a + gridnum   # e.g. 127 = -1 + 128
						if periodic == 0: # if not periodic, dont deposit into this (otherwise wrapped) cell
							fraction = 0.0

					elif (a >= gridnum):
						an = a - gridnum  # e.g. 0 = 128 - 128
						if periodic == 0:
							fraction = 0.0

					if (b < 0):
						bn = b + gridnum
						if periodic == 0:
							fraction = 0.0

					elif (b >= gridnum):
						bn = b - gridnum
						if periodic == 0:
							fraction = 0.0

					if (c < 0):
						cn = c + gridnum
						if periodic == 0:
							fraction = 0.0

					elif (c >= gridnum):
						cn = c - gridnum
						if periodic == 0:
							fraction = 0.0

					for f in range(num_fields):
						fields[an, bn, cn, f] += quantities[n] * fraction
					weights[an, bn, cn] += fraction
	
	return fields, weights


