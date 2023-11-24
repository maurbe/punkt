import numpy as np
import scipy.spatial as spatial
from cython_routines.cython_functions import *


def cic_deposition(positions,
				   quantities,
				   averaged,
				   gridnum,
				   extent,
				   pcellsizesHalf=None,
				   num_nn=None,
				   periodic=1
				   ):
	
	dim = positions.shape[-1]
	assert dim==2 or dim==3, f"Particle positions must be of shape (N, 2) or (N, 3) but found {positions.shape}"

	# cast to float32
	positions  = positions.astype('float32')
	quantities = quantities.astype('float32')
	extent     = extent.astype('float32')

	if pcellsizesHalf is not None and num_nn is not None:
		raise Exception('Either provide pcellsizesHalf or num_nn, but not both.')

	if num_nn is not None:
		# print('Building kd-tree...', flush=True)
		# +1e-8 due to error when pos exactly equals boxsize
		point_tree = spatial.cKDTree(positions, boxsize=extent[1] + 1e-8)
		# [0]=distances; [:, -1] we only need the last NN
		pcellsizesHalf = point_tree.query(x=positions, k=num_nn)[0][:, -1]

	if pcellsizesHalf is not None:
		pcellsizesHalf = pcellsizesHalf.astype('float32')

	if pcellsizesHalf is not None:
		deposition_strategy = cic_2d_adaptive if dim==2 else cic_3d_adaptive
	else:
		deposition_strategy = cic_2d if dim==2 else cic_3d

	if pcellsizesHalf is not None:
		fields, weights = deposition_strategy(positions, 
											  quantities, 
											  pcellsizesHalf, 
											  extent, 
											  gridnum, 
											  periodic)
	else:
		fields, weights = deposition_strategy(positions, 
											  quantities,
											  extent, 
											  gridnum, 
											  periodic)

	# divide averaged fields by weight
	for i in range(len(averaged)):
		if averaged[i] == True:
			fields[..., i] /= (weights + 1e-8)

	return fields, weights



def isotropic_kernel_deposition(positions,
								hsm,
								quantities,
								averaged,
								gridnum,
								extent,
								periodic=1
								):
	
	dim = positions.shape[-1]
	assert dim==2 or dim==3, f"Particle positions must be of shape (N, 2) or (N, 3) but found {positions.shape}"
	
	if len(quantities.shape)==1:
		quantities = quantities[:, np.newaxis]

	# cast to float32
	positions  = positions.astype('float32')
	hsm        = hsm.astype('float32')
	quantities = quantities.astype('float32')
	extent 	   = extent.astype('float32')

	deposition_strategy = isotropic_kernel_deposition_2d if dim==2 else isotropic_kernel_deposition_3d
	fields, weights = deposition_strategy(positions,
										  hsm,
										  quantities,
										  extent, 
										  gridnum, 
										  periodic)

	# divide averaged fields by weight
	for i in range(len(averaged)):
		if averaged[i] == True:
			fields[..., i] /= (weights + 1e-8)

	return fields, weights



def anisotropic_kernel_deposition(positions,
								  hmat,
								  quantities,
								  averaged,
								  gridnum,
								  extent,
								  periodic=1,
								  plane='xy',
								  evals=None,
								  evecs=None,
								  return_evals=False
								  ):
	
	dim = positions.shape[-1]
	assert dim==2 or dim==3, f"Particle positions must be of shape (N, 2) or (N, 3) but found {positions.shape}"
	
	if len(quantities.shape)==1:
		quantities = quantities[:, np.newaxis]

	if dim == 2:
		# need to project the smoothing ellipses on the cartesian plane
		if plane == 'xy':
			e1 = [1, 0, 0]
			e2 = [0, 1, 0]
		elif plane == 'xz':
			e1 = [1, 0, 0]
			e2 = [0, 0, 1]
		elif plane == 'yz':
			e1 = [0, 1, 0]
			e2 = [0, 0, 1]
		else:
			print("Plane must be either xy, yz or xz.")


		# Compute P * M_inverse * P_transpose
		projection_matrix = np.array([e1, e2]).astype('float32')
		hmat_inv = np.linalg.inv(hmat)
		hmat_2d = []

		for i in range(len(hmat_inv)):
		    hmat_2d.append(np.linalg.inv(np.dot(projection_matrix, np.dot(hmat_inv[i], projection_matrix.T))))
		hmat_2d = np.asarray(hmat_2d)

		# compute the new eigenvectors (2d)
		evals, evecs = np.linalg.eigh(hmat_2d)


	# cast to float32
	positions  = positions.astype('float32')
	quantities = quantities.astype('float32')
	extent 	   = extent.astype('float32')
	evals	   = evals.astype('float32')
	evecs	   = evecs.astype('float32')

	# cache it, for some reason the cython functions change the evals and evecs
	if return_evals:
		evals_copy = np.copy(evals)
		evecs_copy = np.copy(evecs)

	deposition_strategy = anisotropic_kernel_deposition_2d if dim==2 else anisotropic_kernel_deposition_3d
	fields, weights = deposition_strategy(positions,
										  evecs,
										  evals,
										  quantities,
										  extent, 
										  gridnum, 
										  periodic)

	# divide averaged fields by weight
	for i in range(len(averaged)):
		if averaged[i] == True:
			fields[..., i] /= (weights + 1e-8)

	return (fields, weights, evals_copy, evecs_copy) if return_evals else (fields, weights)

