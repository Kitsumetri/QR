import numpy as np
import sympy as sp


def qr_decomposition(A):

    def householder(x):
        e = np.array([1 if _ == 0 else 0 for _ in range(x.shape[0])])
        u = x + np.linalg.norm(x) * e
        u = u / np.linalg.norm(u)
        return np.identity(x.shape[0]) - 2 * np.outer(u, u)

    r, c = A.shape
    Q = np.identity(r)
    for i in range(c - (r == c)):
        H = np.identity(r)
        H[i:, i:] = householder(A[i:, i])
        Q = Q @ H
        A = H @ A
    return Q, A


def find_eigvals(A):
    A1 = sp.Matrix(A)
    lamda = sp.symbols('lamda')
    p = A1.charpoly(lamda)
    ans = sp.solvers.solve(p, lamda)
    print('\nSet 2 of eigvals:', *ans, sep='\n')


def main():

    A = np.array([[3, -1, 1, 3],
                  [-1, 2, -1, 0],
                  [1, 0, 1, 1],
                  [1, 2, -1, 4]])

    # A = np.array([[1, 1, 5],
    #               [2, 1, 2],
    #               [3, 1, -5]])

    # A = np.array([[5, -1, -1],
    #               [0, 4, -1],
    #               [0, -1, 4]])

    # A = np.array([[1, -1, 0],
    #               [-1, 0, 1],
    #               [0, 1, 1]])

    A1 = np.copy(A)

    for i in range(2):
        # q, r = qr_decomposition(A1)
        q, r = np.linalg.qr(A1, mode='complete')
        A1 = r @ q

    print('ans =\n', A1.round(4), '\n')

    if np.allclose(A1.round(1), np.triu(A1).round(1)):
        print('Eigvals:', *np.diag(A1), sep='\n')
    else:
        vals = np.diag(A1)[2:]
        print('Set 1 of eigvals:', *vals, sep='\n')
        find_eigvals(A1[0:2, :2])

    print('\nValues using numpy: ', *np.linalg.eigvals(A), sep='\n')


if __name__ == '__main__':
    main()
