"""
This class provides the estimation for the parameters
M_true and w_true for the quadratic problem.

Note that M_true is rank-2 symmetric.

Input: Feature matrix X, and label y

Where: y_i = x_i^T M_true x_i + x_i^T w_true
"""

import numpy as np
from scipy.linalg import svd, qr
import matplotlib.pyplot as plt

from typing import List

class Sensing:
    """
    The class Sensing works as an optimizer to estimate M_true and w_true
    """

    def __init__(self, B: int, N: int, r: int=2) -> None:
        """
        Initialization.

        Args:
            B (int): batch size
            N (int): dimension of the symmetric M, in our case, it is the number
                     of reflective elements
            r (int, optional): rank of matrix M_true. Defaults to 2.
                     In theory, rank of M_true can be 1. But in reality, that will
                     never happen.
        """
        self.B = B
        self.N = N
        self.r = r

    def solve(self, X: np.array, y: np.array,
              err_var: float=10**-3, convergence_plot: bool=False) -> List:
            """
            Main function to perform sensing.

            Args:
                X (np.array): Feature data
                y (np.array): Observation/label data
                convergence_plot (bool, optional): Plot the convergence graph.
                                                   Defaults to False.
                err_var (float, optional): The algorithm will stop if the variance
                    of estimations is lower than err_var. Defaults to 10**-3.

            Returns:
                the estimated M_true and w_true
                all_norms: |\mathbf{M}\|_2 + \|\mathbf{w}\|_2 over iterations
            """
            B, N, r = self.B, self.N, self.r
            
            def compute_A(X, M):
                # A_i(M) = x_i.T @ M @ x_i
                # shape of X: N x B
                # A(M) should be a column vector, shape B x 1
                B = X.shape[1]
                return np.array([X[:, i].T @ M @ X[:, i] for i in range(B)]).reshape(-1, 1)

            def compute_A_adj(z, X):
                # compute adjoint operator = 1/2/B * sum (z_i * x_i * x_i.T)
                # z is a column vector
                # return a matrix whose shape N x N
                B = X.shape[1]
                sum = 0
                for i in range(B):
                    sum += z[i, 0] * X[:, i].reshape(-1, 1) @ X[:, i].reshape(-1, 1).T
                return sum
            
            # Initialization using the first mini batch
            w = np.zeros((N, 1))
            V = np.zeros((N, r))
            X_init = X[:, :B]
            y_init = y[:B]
            M_init = np.zeros((N, N))
            A_M_init = compute_A(X=X_init, M=M_init)
            r_vec = y_init - A_M_init - X_init.T @ w
            H1_init = compute_A_adj(r_vec, X_init)
            h2_init = np.mean(r_vec)
            matrix_init = H1_init - 0.5 * h2_init * np.eye(N)
            U_init, _, _ = svd(matrix_init)
            U = U_init[:, :r]

            all_norms = [] # store the norm of M + the norm of w, to confirm convergence
            i = 0
            perm_count = 0 # count how many times we shuffle the data
            # compute each mini-batch
            while True:
                # current batch data
                start = i * B
                end = start + B
                X_t = X[:, start:end]
                y_t = y[start:end]
                
                # this is the current M computed using U and V from previous batch
                M = 0.5 * (U @ V.T + V @ U.T)
                
                A_M = compute_A(X=X_t, M=M)
                r_t = y_t - A_M - X_t.T @ w

                H1 = 1/2/B * compute_A_adj(z=r_t, X=X_t)
                h2 = np.mean(r_t)
                h3 = 1/B * X_t @ r_t

                matrix_update = H1 - 0.5 * h2 * np.eye(N) + M.T
                U_hat = matrix_update @ U
                U, _ = qr(U_hat, mode='economic')

                w = w + h3
                V = matrix_update @ U

                # store the sum of norm
                all_norms.append(np.linalg.norm(M) + np.linalg.norm(w))
                i += 1
                if i * B + B > len(y):
                    perm = np.random.permutation(len(y))
                    perm_count += 1
                    X = X[:, perm]
                    y = y[perm]
                    i = 0
                var = np.var(all_norms[-5:])
                if  var > 0 and var < err_var:
                    break
                if perm_count == 10:
                    print(f'Variance of the last 5 estimates = {var}')
                    print('Consider to increase the batch size or number of samples.')
                    break
                    
            if convergence_plot:
                plt.plot(all_norms, 'red', marker='.', lw = 0.5)
                plt.xlabel('iteration', {'fontname':'Helvetica'})
                plt.ylabel(r'$\|\mathbf{M}\|_2 + \|\mathbf{w}\|_2$')
                plt.grid()
                plt.show()
        
            return M, w, all_norms
    
    def project_to_psd_lowrank(self, M: np.array, r: int) -> np.array:
        """
        Projects a symmetric matrix M onto the set of positive semi-definite matrices
        with rank at most r, minimizing the Frobenius norm distance.
        
        Parameters:
            M (np.ndarray): Symmetric matrix of shape (N, N).
            r (int): Maximum rank for the projection.
        
        Returns:
            np.ndarray: The projected matrix M^+.
        """
        N = M.shape[0]
        if r < 0 or r > N:
            raise ValueError("r must be between 0 and N")
        
        # Compute eigendecomposition; eigh returns ascending eigenvalues
        vals, vecs = np.linalg.eigh(M)
        
        # Sort eigenvalues in descending order
        idx = np.argsort(vals)[::-1]
        sorted_vals = vals[idx]
        V = vecs[:, idx]  # Eigenvectors corresponding to descending eigenvalues
        
        # Compute the diagonal for D^+
        diag_plus = np.zeros(N)
        diag_plus[:r] = np.maximum(0, sorted_vals[:r])
        
        # Reconstruct M^+
        M_plus = V @ np.diag(diag_plus) @ V.T
        
        return M_plus
