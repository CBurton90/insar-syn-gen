import numpy as np



alpha0 = 0.9 * np.max(Mx)
lmbda = 1 # eigenvalue of (M**-1 M) -> eigenvalue of Identity matrix?

while alpha > (tol * alpha0):
    for k in range(K):
        x = x + (1/lmbda * np.linalg.inv(M) * (y - )
