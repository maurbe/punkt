import numpy as np
from numpy.fft import fftn, fftfreq, ifftn


def PeriodicPoissonSolver(rhs, dx=1):
	# https://github.com/tp5uiuc/py_poisson_solver/blob/3e916ab9ba677fd876d88eb29756b3c85dc7fc60/poisson_solver/periodic_poisson_solver.py#L10

	grid_size = len(rhs)
	freqs = np.fft.fftfreq(grid_size)
	kx, ky, kz = np.meshgrid(freqs, freqs, freqs)

	laplacian_in_fourier_domain = (-4 * np.pi ** 2 * (kx ** 2 + ky ** 2 + kz ** 2) / dx / dx / dx)
	laplacian_in_fourier_domain[0, 0, 0] = np.inf

	sol = ifftn(fftn(rhs) / laplacian_in_fourier_domain)
	return sol.real


def compute_web(mass_grid, thresh=0.2):
	
	# deposit mass on grid and compute overdensity delta
	#mass_on_grid = cic_deposition(positions, mass, gridnum, extent, periodic=periodic)
	delta = mass_grid / mass_grid.mean() - 1.0

	# solve poisson equation laplace(phi) = delta
	print('Computing T...')
	phi = PeriodicPoissonSolver(delta)

	# compute deformation tensor
	grads = np.gradient(phi)
	hessian = np.empty(phi.shape + (phi.ndim, phi.ndim), dtype=phi.dtype)
	for k, grad_k in enumerate(grads):
		tmp_grad = np.gradient(grad_k)
		for l, grad_kl in enumerate(tmp_grad):
			hessian[..., k, l] = grad_kl

	# compute eigenvalues of deformation tensor
	print('Classifying web...')
	eigvals = np.linalg.eigh(hessian)[0]
	web = np.sum(eigvals > thresh, axis=-1)

	return web
