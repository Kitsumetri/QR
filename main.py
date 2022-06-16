import numpy as np
import sympy as sp


def qr_decomposition(A):

    def householder(x):
        e = np.array([1 if _ == 0 else 0 for _ in range(x.shape[0])])
        u = x + np.linalg.norm(x) * e
        u /= np.linalg.norm(u)
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
    return sp.solvers.solve(p, lamda)


def get_bad_eigvals(A, e):
    values = set()
    i = 0
    while i < A.shape[0] - 1:
        if A[i + 1][i] < e:
            values.add(A[i][i])
            i += 1
        else:
            roots = find_eigvals(A[i:i+2, i:i+2])
            for root in roots:
                values.add(root)
            i += 2
    else:
        if i == A.shape[0] - 1:
            values.add(A[i][i])

    return tuple(values)


def get_bad_eigvals_v2(A, e):
    values = set()
    i = 0
    while i < A.shape[0] - 1:
        j = 1
        if A[i + 1][i] < e:
            values.add(A[i][i])
            i += 1
        else:
            index = i
            while index < A.shape[0] - 1 and A[index + 1][index] >= e:
                j += 1
                index += 1

            roots = find_eigvals(A[i:i+j, i:i+j])
            for root in roots:
                values.add(root)
            i += 2
    else:
        if i == A.shape[0] - 1:
            values.add(A[i][i])

    return tuple(values)


def main():

    # A = np.array([[3, -1, 10, 3],
    #               [-1, -10, -1, 0],
    #               [1, 0, 2.3, 1],
    #               [-49, 2, -1, -5]])

    # A = np.array([[1, 3, 4, 5, 3],
    #               [2, 1, 9, 3, 4],
    #               [7, 3, 2, 5, 8],
    #               [6, 3, 1, 0, 8],
    #               [1, 4, 2, 6, 9]])

    # A = np.array([[1, 1, 5],
    #               [2, 1, 2],
    #               [3, 1, -5]])

    # A = np.array([[1, -2, 3],
    #               [4, 5, -6],
    #               [-7, 8, 9]])

    # A = np.array([[5, -1, -1],
    #               [0, 4, -1],
    #               [0, -1, 4]])

    A = np.array([[1, -1, 0],
                  [-1, 0, 1],
                  [0, 1, 1]])

    A1 = np.copy(A)

    for i in range(200):
        q, r = qr_decomposition(A1)
        A1 = r @ q

    print('Answer matrix =\n', A1, '\n')

    if np.allclose(A1, np.triu(A1)):
        print('Eigvals:\n', *np.diag(A1))
    else:
        print('Eigvals:\n', *get_bad_eigvals_v2(A1, 1e-6), sep='\n')

    print('\nValues using numpy: ', *np.linalg.eigvals(A), sep='\n')


if __name__ == '__main__':
    main()
