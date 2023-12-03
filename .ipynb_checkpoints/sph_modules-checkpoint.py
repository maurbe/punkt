import math
import numpy as np
from scipy.spatial import cKDTree


class particle2mesh():
    
    def __init__(self,
                 particle_masses,
                 boxsize,
                 number_of_nn,
                ):
        
        self.pmasses = particle_masses
        self.boxsize = boxsize
        self.nn      = number_of_nn
    
        # compute relevant quantities for particles
        # the particle density must always be computed in 3d
        # print('Init: \t computing 3d hsm and ρ')
    
    
    def setup(self, dim, pos=None):
        # if we want to do a 2d projection -> recompute the kernel sizes for particles
        # but the density must be kept 3d
        
        self.hsm, nn_dists, nn_inds, self.tree = self._compute_hsm(pos)
        self.rho = self._compute_density(dim=dim, hsm=self.hsm, nn_dists=nn_dists, nn_inds=nn_inds)
        self.ks = self.hsm
        
        return self
    
    
    def _compute_hsm(self, pos):
        """
        pos:           particle positions
        nrn:           number of neighbors to consider
        returns:       computes the smoothing length for each 
                       particle as 0.5 * distance to Nth nn.
        """
        
        # k+1 -> do not consider particle itself
        tree = cKDTree(pos, boxsize=self.boxsize)
        nn_dists, nn_inds = tree.query(pos, k=self.nn+1)

        # Note: particle itself is really not used when computing its density
        hsm       = nn_dists[:, -1] * 0.5
        nn_dists  = nn_dists[:, 1:]
        nn_inds   = nn_inds[:, 1:]
        
        return hsm, nn_dists, nn_inds, tree
    
    
    def _compute_density(self, dim, hsm, nn_dists, nn_inds):
        
        w_ij = self._quintic_spline(dim, hsm, nn_dists)
        ρ_i  = np.sum( self.pmasses[nn_inds] * w_ij , axis=1)
        return ρ_i
    
    
    def _quintic_spline(self, dim, h, r_ij):
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
    
    
    def query_A_at_positions(self, 
                             particle_A,
                             pos_interpol
                             ):
        """
        A:             vector of A at particle positions
        masses:        particle masses
        densities:     particle densites
        hsm:           particle smoothing lengths
        tree:          ckdtree
        pos_interpol:  positions at which to compute the field value A

        returns:       interpolated quantity A at query positions
        """
        print('Query: \t querying grid')
        dim = pos_interpol.shape[-1]
        
        # the "h" in the formula is the smoothing length of particle j
        # no need to recompute new smoothing lengths for "virtual particle" query positions
        nn_dists, nn_inds = self.tree.query(pos_interpol, k=self.nn)
        
        # average of hsm's ij, i.e. "gather-scatter" approach
        h_i = (nn_dists[:, -1] * 0.5)[:, np.newaxis]
        h_j = self.ks[nn_inds]
        h_ij= (h_i + h_j) * 0.5

        w_ij= self._quintic_spline(dim=dim, h=h_ij, r_ij=nn_dists)
        A_i = np.sum( particle_A[nn_inds] * w_ij * self.pmasses[nn_inds] / self.rho[nn_inds], axis=1)

        return A_i
    
    
    def create_grid_1d(self, nx):

        Δx = self.boxsize / nx
        x = np.linspace(Δx / 2.0, self.boxsize - Δx/2.0, nx)
        x = x[:, np.newaxis]
        return x.astype('float32')
    
    
    def create_grid_2d(self, nx, ny):

        Δx = self.boxsize / nx
        Δy = self.boxsize / ny

        x = np.linspace(Δx / 2.0, self.boxsize - Δx/2.0, nx)
        y = np.linspace(Δy / 2.0, self.boxsize - Δy/2.0, ny)

        xx, yy = np.meshgrid(x, y, indexing='ij')
        grid_positions = np.stack((xx.ravel(), 
                                   yy.ravel()), axis=-1).astype('float32')
        return grid_positions
    
    
    def create_grid_3d(self, nx, ny, nz):

        Δx = self.boxsize / nx
        Δy = self.boxsize / ny
        Δz = self.boxsize / nz

        x = np.linspace(Δx / 2.0, self.boxsize - Δx/2.0, nx)
        y = np.linspace(Δy / 2.0, self.boxsize - Δy/2.0, ny)
        z = np.linspace(Δz / 2.0, self.boxsize - Δz/2.0, nz)

        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        grid_positions = np.stack((xx.ravel(), 
                                   yy.ravel(), 
                                   zz.ravel()), 
                                  axis=-1)
        return grid_positions
