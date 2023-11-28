import math
import numpy as np
from scipy.spatial import cKDTree


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
		self.rho = self._compute_density(dim=dim, hsm=self.hsm, nn_dists=nn_dists, nn_inds=nn_inds)		

	
	def _compute_density(self, dim, hsm, nn_dists, nn_inds):
		
		w_ij = quintic_spline(dim, hsm, nn_dists)
		ρ_i  = np.sum( self.pmasses[nn_inds] * w_ij , axis=1)
		return ρ_i
	
	
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


def compute_hsm(pos, nn, boxsize):
	"""
	pos:           particle positions
	nrn:           number of neighbors to consider
	returns:       computes the smoothing length for each 
				   particle as 0.5 * distance to Nth nn.
	"""
	
	# k+1 -> do not consider particle itself
	tree = cKDTree(pos, boxsize=boxsize)
	nn_dists, nn_inds = tree.query(pos, k=nn+1)

	# Note: particle itself is really not used when computing its density
	hsm       = nn_dists[:, -1] * 0.5
	nn_dists  = nn_dists[:, 1:]
	nn_inds   = nn_inds[:, 1:]
	
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

	# eigenvectors are returned normalized
	# eigvecs are the same for H
	# eigvals are the sqrt of the ones of Sigma
	eigvals, eigvecs = np.linalg.eigh(Sigma)
	eigvals = np.sqrt(eigvals)
	
	# also compute H = VΛV.T
	Λ = eigvals[..., np.newaxis] * np.eye(pos.shape[-1])
	H = np.matmul(np.matmul(eigvecs, Λ), np.transpose(eigvecs, axes=(0, 2, 1)))
	
	return H, eigvals, eigvecs


def quintic_spline(dim, h, r_ij):
	"""
	h:             smoothing lengths
	q:             abs(relative coords)
	"""
	if len(h.shape)==1:
		h = h[:, np.newaxis]

	if dim == 1:
		sigma = 1.0 / (120 * h)
	elif dim == 2:
		sigma = 7.0 / (478 * math.pi * h ** 2)
	elif dim == 3:
		sigma = 1.0 / (120 * math.pi * h ** 3)

	q = r_ij / h
	term = np.zeros_like(q)
	term = np.where(np.logical_and(0<=q, q<=1), (3-q)**5 - 6*(2-q)**5 + 15*(1-q)**5, term)
	term = np.where(np.logical_and(1< q, q<=2), (3-q)**5 - 6*(2-q)**5, term)
	term = np.where(np.logical_and(2< q, q<=3), (3-q)**5, term)

	W = sigma * term
	return W

	
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