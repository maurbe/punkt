# +
import math
import numpy as np

import scipy.spatial as spatial
from cython_routines.cython_functions import *


# -

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
			fields[..., i] /= (weights + 1e-10)

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
	gridnum	   = int(gridnum)

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
			fields[..., i] /= (weights + 1e-10)

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
	gridnum	   = int(gridnum)

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
			fields[..., i] /= (weights + 1e-10)

	return (fields, weights, evals_copy, evecs_copy) if return_evals else (fields, weights)


# +
class sph_interpolator():
	
	def __init__(self,
				 particle_positions,
				 particle_masses,
				 boxsize,
				 number_of_nn
				 ):
		
		self.pmasses = particle_masses
		self.nn      = number_of_nn
		dim = particle_positions.shape[-1]

		# compute relevant quantities for particles
		print(f'Init: \t computing {dim}d hsm and ρ')
		self.hsm, nn_dists, nn_inds, self.tree = compute_hsm(particle_positions, self.nn, boxsize)
		self.rho = compute_density(dim=dim, hsm=self.hsm, masses=self.pmasses, nn_dists=nn_dists, nn_inds=nn_inds)		
	
	
	def query_field_at_positions(self, 
								 particle_field,
								 pos_interpol
								 ):
		"""
		particle_field: vector of quantity at particle positions
		masses:         particle masses
		densities:      particle densites
		hsm:            particle smoothing lengths
		tree:           ckdtree
		pos_interpol:   positions at which to compute the field value A

		returns:        interpolated quantity A at query positions
		"""
		print('Query: \t querying values at coordinates')
		dim = pos_interpol.shape[-1]
		
		# the "h" in the formula is the smoothing length of particle j
		# no need to recompute new smoothing lengths for "virtual particle" query positions
		nn_dists, nn_inds = self.tree.query(pos_interpol, k=self.nn)
		
		# average of hsm's ij, i.e. "gather-scatter" approach
		h_i = (nn_dists[:, -1] * 0.5)[:, np.newaxis]
		h_j = self.hsm[nn_inds]
		h_ij= (h_i + h_j) * 0.5

		w_ij= quintic_spline(dim=dim, h=h_ij, r_ij=nn_dists)
		A_i = np.sum( particle_field[nn_inds] * w_ij * self.pmasses[nn_inds] / self.rho[nn_inds], axis=1)

		return A_i


def compute_hsm(pos, nn, boxsize=None):
	"""
	pos:           	particle positions
	nn:            	number of neighbors to consider
	boxsize:		if specified, use periodic kdtree, else non-periodic
	returns:       	computes the smoothing length for each 
				   	particle as 0.5 * distance to Nth nn.
	"""
	
	# k+1 -> do not consider particle itself
	if boxsize is None:
		tree = spatial.KDTree(pos)	
	else:
		tree = spatial.cKDTree(pos, boxsize=boxsize)
	nn_dists, nn_inds = tree.query(pos, k=nn)

	# compute hsm
	hsm = nn_dists[:, -1] * 0.5
	return hsm, nn_dists, nn_inds, tree


def compute_hsm_tensor(pos, masses, NN, boxsize):

	# use compute_hsm() to find the nn_inds
	_, _, nn_inds, _ = compute_hsm(pos, NN, boxsize)
	
	neighbor_coords = pos[nn_inds]
	neighbor_masses = masses[nn_inds]
	
	# we have to account for pbc
	r_jc = neighbor_coords - pos[:, np.newaxis, :]
	r_jc = np.where(np.abs(r_jc) >= boxsize / 2.0, r_jc - np.sign(r_jc) * boxsize, r_jc)
	
	outer = np.einsum('...i, ...j -> ...ij', r_jc, r_jc)
	outer = outer * neighbor_masses[..., np.newaxis, np.newaxis]
	Sigma = np.sum(outer, axis=1) / np.sum(neighbor_masses, axis=1)[..., np.newaxis, np.newaxis]

	# eigvecs are returned normalized
	# eigvecs are the same for H
	# eigvals are the sqrt of the ones of Sigma
	eigvals, eigvecs = np.linalg.eigh(Sigma)
	eigvals = np.sqrt(eigvals)
	
	# also compute H = VΛV.T
	Λ = eigvals[..., np.newaxis] * np.eye(pos.shape[-1])
	H = np.matmul(np.matmul(eigvecs, Λ), np.transpose(eigvecs, axes=(0, 2, 1)))
	
	return H, eigvals, eigvecs


def compute_div(masses, rho, A, nn_inds, grad_w):
	
	# need to ignore particle itself for gradient computation
	nn_inds = nn_inds[:, 1:]
	A_j = A[nn_inds]
	
	div_A  = 1 / rho[:, np.newaxis] * np.sum( masses[nn_inds][..., np.newaxis] * (A_j[..., np.newaxis] - A[:, np.newaxis, np.newaxis]) * grad_w, axis=1)
	return div_A


def compute_rot(masses, rho, A, nn_inds, grad_w):
	
	# need to ignore particle itself for gradient computation
	nn_inds = nn_inds[:, 1:]
	A_j = A[nn_inds]
	
	rot_A  = 1 / rho[:, np.newaxis] * np.sum( masses[nn_inds][..., np.newaxis] * np.cross(A_j - A[:, np.newaxis], grad_w), axis=1)
	return rot_A


def compute_density(dim, hsm, masses, nn_dists, nn_inds):
	"""
	dim:           how many spatial dims
	hsm:           particle smoothing lengths
	masses:		   particle masses
	nn_dists:	   distances to neighbors
	nn_inds:       list of particle neighbors
	returns:       computes the density at particle positions
	"""
	w_ij = quintic_spline(dim, hsm, nn_dists)
	ρ_i  = np.sum( masses[nn_inds] * w_ij , axis=1)
	return ρ_i


def compute_vsm(vel, nn_inds):
	"""
	vel:           particle velocities
	nn_inds:       list of particle neighbors
	returns:       computes the velocity dispersion vector
	"""
	vel_nn = vel[nn_inds]
	#vel_nn = np.concatenate([vel_nn, vel[:, np.newaxis]], axis=1).mean(axis=1) # attach the particle vel itself
	vsm = vel - vel_nn.mean(axis=1)
	return vsm


def σ5(dim, h):

	if dim == 1:
		sigma = 1.0 / (120 * h)
	elif dim == 2:
		sigma = 7.0 / (478 * math.pi * h ** 2)
	elif dim == 3:
		sigma = 1.0 / (120 * math.pi * h ** 3)

	return sigma
	

def quintic_spline(dim, h, r_ij):
	"""
	dim:		   spatial dim
	h:             smoothing lengths
	q:             abs(relative coords)
	"""
	if len(h.shape)==1:
		h = h[:, np.newaxis]
	sigma = σ5(dim, h)

	q = r_ij / h
	term = np.zeros_like(q)
	term = np.where(np.logical_and(0<=q, q<=1), (3-q)**5 - 6*(2-q)**5 + 15*(1-q)**5, term)
	term = np.where(np.logical_and(1< q, q<=2), (3-q)**5 - 6*(2-q)**5, term)
	term = np.where(np.logical_and(2< q, q<=3), (3-q)**5, term)

	W = sigma * term
	return W


def quintic_spline_gradient(dim, pos, h, nn_inds):
	"""
	dim:           spatial dim
	pos:   	       particle positions
	h:             smoothing lengths
	nn_inds:       indices of neighbors
	"""
	# need to ignore particle itself in the gradient computation (would give division by 0 error)
	nn_inds = nn_inds[:, 1:]
	
	# prepare normalizations
	if len(h.shape)==1:
		h = h[:, np.newaxis]
	sigma = σ5(dim, h)
	
	r_ij_vec = pos[nn_inds] - pos[:, np.newaxis, :] # this holds un-normalized vector differences!
	r_ij_mag = np.linalg.norm(r_ij_vec, axis=-1)

	er = r_ij_vec / r_ij_mag[..., np.newaxis]
	q = r_ij_mag / h

	dwdq = np.zeros_like(q)
	dwdq = np.where(np.logical_and(0<=q, q<=1), -5 * (3-q)**4 + 5*6*(2-q)**4 - 5*15*(1-q)**4, dwdq)
	dwdq = np.where(np.logical_and(1< q, q<=2), -5 * (3-q)**4 + 5*6*(2-q)**4, dwdq)
	dwdq = np.where(np.logical_and(2< q, q<=3), -5 * (3-q)**4, dwdq)
	dwdr = sigma / h * dwdq
	
	dwdx = dwdr * er[..., 0]
	dwdy = dwdr * er[..., 1]
	if dim == 3:
		dwdz = dwdr * er[..., 2]
		grad = np.stack([dwdx, dwdy, dwdz], axis=-1)

	elif dim == 2:
		grad = np.stack([dwdx, dwdy], axis=-1)

	return grad


def create_grid_1d(nx, boxsize):

	Δx = boxsize / nx
	x = np.linspace(Δx / 2.0, boxsize - Δx/2.0, nx)
	x = x[:, np.newaxis]
	return x.astype('float32')


def create_grid_2d(nx, ny, boxsize):

	Δx = boxsize / nx
	Δy = boxsize / ny

	x = np.linspace(Δx / 2.0, boxsize - Δx/2.0, nx)
	y = np.linspace(Δy / 2.0, boxsize - Δy/2.0, ny)

	xx, yy = np.meshgrid(x, y, indexing='ij')
	grid_positions = np.stack((xx.ravel(), 
							   yy.ravel()), axis=-1).astype('float32')
	return grid_positions


def create_grid_3d(nx, ny, nz, boxsize):

	Δx = boxsize / nx
	Δy = boxsize / ny
	Δz = boxsize / nz

	x = np.linspace(Δx / 2.0, boxsize - Δx/2.0, nx)
	y = np.linspace(Δy / 2.0, boxsize - Δy/2.0, ny)
	z = np.linspace(Δz / 2.0, boxsize - Δz/2.0, nz)

	xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
	grid_positions = np.stack((xx.ravel(), 
							   yy.ravel(), 
							   zz.ravel()), 
							  axis=-1)
	return grid_positions



"""
def grad_W(dim, h, r_i, r_js):
	#h:             smoothing lengths
	#q:             abs(relative coords)
 
	if len(h.shape)==1:
		h = h[:, np.newaxis]

	if dim == 1:
		sigma = 1.0 / (120 * h)
	elif dim == 2:
		sigma = 7.0 / (478 * math.pi * h ** 2)
	elif dim == 3:
		sigma = 1.0 / (120 * math.pi * h ** 3)
	
	for j in range(r_js.shape[-2]):
		r_js[:, j] = r_i - r_js[:, j]

	x_ij = r_js # this holds vector differences!
	r_ij = np.linalg.norm(x_ij, axis=-1)
	q = r_ij / h
	
	dwdq = np.zeros_like(q)
	dwdq = np.where(np.logical_and(0<=q, q<=1), (-5) * (3-q)**4 + 5*6*(2-q)**4 - 5*15*(1-q)**4, dwdq)
	dwdq = np.where(np.logical_and(1< q, q<=2), (-5) * (3-q)**4 + 5*6*(2-q)**4, dwdq)
	dwdq = np.where(np.logical_and(2< q, q<=3), (-5) * (3-q)**4, dwdq)
	dwdq = sigma * dwdq
	
	dwdq = dwdq / (h * r_ij)
	dwdx = dwdq * x_ij[..., 0]
	dwdy = dwdq * x_ij[..., 1]
	
	if dim == 3:
		dwdz = dwdq * x_ij[..., 2]
		grad = np.stack([dwdx, dwdy, dwdz], axis=-1)
	
	elif dim == 2:
		grad = np.stack([dwdx, dwdy], axis=-1)

	return grad

def grad_rho(rho, masses, grad_w, nn_inds):
	rho_i = rho[:, np.newaxis, np.newaxis]
	rho_j = rho[nn_inds][..., np.newaxis]
	m_j = masses[:, np.newaxis, np.newaxis]
	return np.sum( m_j * (rho_j - rho_i) / rho_j * grad_w, axis=-2)

def grad_rho2(rho, grad_rho, masses, grad_w, nn_inds):
	
	m_j = masses[nn_inds]
	m_j = m_j[..., np.newaxis, np.newaxis]
	
	#gradW_j = grad_w[nn_inds][..., np.newaxis]
	print(grad_w.shape)
	
	rho_i = rho[..., np.newaxis, np.newaxis, np.newaxis]
	rho_j = rho[nn_inds][..., np.newaxis, np.newaxis]
	
	grad_rho_i = grad_rho[..., np.newaxis, :, np.newaxis]
	grad_rho_j = grad_rho[nn_inds][..., np.newaxis, :]
	
	print(rho_i.shape, rho_j.shape)
	print(grad_rho_i.shape, grad_rho_j.shape)
	
	print((grad_rho_i / rho_j).shape, (grad_rho_j / rho_i).shape)
	
	outer_prod = np.matmul(grad_rho_i / rho_j, 
						   grad_rho_j / rho_i)
	print(outer_prod.shape)
	

	tensor = np.sum(m_j * outer_prod, axis=1)
	
	# compute three eigenvectors
	eigvals, eigvectors = np.linalg.eigh(tensor)
	print(eigvals.shape, eigvectors.shape)
	return eigvectors
	#return np.sum(m_j * outer_prod * gradW_j, axis=1)

h_sphere, nn_dists, nn_inds, tree = _compute_hsm(pos, boxsize, NN)
density = _compute_density(2, h_sphere, masses, nn_dists, nn_inds)
grad    = grad_W(dim=2, h=h_sphere, r_i=pos, r_js=pos[nn_inds])

rho_grad = grad_rho(density, masses, grad, nn_inds)
eigenvectors = grad_rho2(density, rho_grad, masses, grad, nn_inds)

print(eigenvectors.shape)

# re-scale the eigenvectors with hsm
eigenvectors = eigenvectors * h_sphere[..., np.newaxis, np.newaxis]
"""
