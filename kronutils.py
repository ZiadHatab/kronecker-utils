"""
Author: Ziad (https://github.com/ziadhatab)

A collection of utility functions for matrix vectorization and Kronecker products.
"""

import numpy as np

def block_diag(*matrices):
    """
    Construct a block diagonal matrix from the input matrices.
    Also known as the direct sum.

    Parameters
    ----------
    *matrices : np.ndarray
        The matrices to be placed on the diagonal.
    
    Returns
    -------
    np.ndarray
        The block diagonal of the input matrices.

    Example
    -------
    >>> A = np.array([[1, 2], [3, 4]])
    >>> B = np.array([[5, 6], [7, 8]])
    >>> block_diag(A, B)
    array([ [1, 2, 0, 0],
            [3, 4, 0, 0],
            [0, 0, 5, 6],
            [0, 0, 7, 8]])
    
    References
    ----------
    - https://en.wikipedia.org/wiki/Direct_sum
    """
    matrices = [np.atleast_2d(mat) for mat in matrices]
    return np.block([
        [matrices[j] if i == j else np.zeros((matrices[i].shape[0], matrices[j].shape[1]), dtype=matrices[i].dtype)
         for j in range(len(matrices))]
        for i in range(len(matrices))
    ])

def extract_blocks(A, block_size_row, block_size_col):
    """
    Extract blocks from a matrix into a 2D list.
    e.g., A = [[A11, A12], [A21, A22]] -> blocks[i][j] gives block Aij

    Parameters
    ----------
    A : np.ndarray
        The matrix to be block extracted.
    block_size_row : list of int
        The height of each row block. len(block_size_row) is the number of row blocks.
    block_size_col : list of int
        The width of each column block. len(block_size_col) is the number of column blocks.

    Returns
    -------
    list of list of np.ndarray
        A 2D list where blocks[i][j] is the block at row i, column j.
    """
    if sum(block_size_row) != A.shape[0]:
        raise ValueError(f"Sum of block_size_row ({sum(block_size_row)}) must equal number of rows ({A.shape[0]}).")
    if sum(block_size_col) != A.shape[1]:
        raise ValueError(f"Sum of block_size_col ({sum(block_size_col)}) must equal number of columns ({A.shape[1]}).")
    m = len(block_size_row)
    n = len(block_size_col)
    row_offsets = np.cumsum(np.hstack([[0], block_size_row]))
    col_offsets = np.cumsum(np.hstack([[0], block_size_col]))
    return [[A[row_offsets[i]:row_offsets[i+1], col_offsets[j]:col_offsets[j+1]] for j in range(n)] for i in range(m)]

def extract_diag_blocks(A, block_size_row, block_size_col):
    """
    Extract diagonal blocks from a matrix into a list.
    e.g., A = [[A11, A12], [A21, A22]] -> diag_blocks[i] gives block Aii

    Parameters
    ----------
    A : np.ndarray
        The matrix to be diagonal block extracted.
    block_size_row : list of int
        The height of each row block. len(block_size_row) is the number of row blocks.
    block_size_col : list of int
        The width of each column block. len(block_size_col) is the number of column blocks.

    Returns
    -------
    list of np.ndarray
        A list where diag_blocks[i] is the diagonal block at row i, column i.
    """
    A_blocks = extract_blocks(A, block_size_row, block_size_col)
    return [A_blocks[i][i] for i in range(min(len(block_size_row), len(block_size_col)))]

def vec(A):
    """
    Column-wise vectorization of a matrix.

    Parameters
    ----------
    A : np.ndarray or list
        The matrix to be vectorized.

    Returns
    -------
    np.ndarray
        The vectorized form of the input matrix.

    References
    ----------
    - https://en.wikipedia.org/wiki/Vectorization_(mathematics)
    - K. B. Petersen and M. S. Pedersen, “The Matrix Cookbook.” 
    Technical University of Denmark, Nov. 2012. 
    [Online]. Available: http://www2.compute.dtu.dk/pubdb/pubs/3274-full.html
    - J. Brewer, "Kronecker products and matrix calculus in system theory," 
    in IEEE Transactions on Circuits and Systems, vol. 25, no. 9, pp. 772-781, 
    September 1978, doi: 10.1109/TCS.1978.1084534.
    """
    A = np.atleast_2d(A)
    return A.flatten(order='F')

def unvec(v, shape):
    """
    Inverse of column-wise vectorization. Reshapes a vector back into a matrix.

    Parameters
    ----------
    v : np.ndarray or list
        The vector to be unvectorized.
    shape : tuple of int
        The shape (rows, cols) of the output matrix.

    Returns
    -------
    np.ndarray
        The unvectorized matrix of the specified shape.

    References
    ----------
    - https://en.wikipedia.org/wiki/Vectorization_(mathematics)
    - K. B. Petersen and M. S. Pedersen, “The Matrix Cookbook.” 
    Technical University of Denmark, Nov. 2012. 
    [Online]. Available: http://www2.compute.dtu.dk/pubdb/pubs/3274-full.html
    - J. Brewer, "Kronecker products and matrix calculus in system theory," 
    in IEEE Transactions on Circuits and Systems, vol. 25, no. 9, pp. 772-781, 
    September 1978, doi: 10.1109/TCS.1978.1084534.
    """
    return np.reshape(v, shape, order='F')

def vech(A):
    """
    Half-vectorization of a symmetric matrix.

    Parameters
    ----------
    A : np.ndarray or list
        The symmetric matrix to be half-vectorized.

    Returns
    -------
    np.ndarray
        The half-vectorized form of the input symmetric matrix.
    
    References
    ----------
    - https://en.wikipedia.org/wiki/Vectorization_(mathematics)
    - K. G. Jinadasa, “Applications of the matrix operators vech and vec,” 
    Linear Algebra and its Applications, vol. 101, pp. 73–79, 1988, 
    doi: https://doi.org/10.1016/0024-3795(88)90143-7.
    - D. Nagakura, “Further results on the vecd operator and its applications,” 
    Communications in Statistics - Theory and Methods, vol. 49, no. 10, pp. 2321–2338, 2020, 
    doi: 10.1080/03610926.2019.1570265.
    """
    A = np.atleast_2d(A)    
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"vech requires a square matrix, but got shape {A.shape}.")    
    return A[np.tril_indices_from(A)]

def unvech(v, N):
    """
    Inverse of half-vectorization. Reshapes a vector back into a symmetric matrix.

    Parameters
    ----------
    v : np.ndarray or list
        The vector to be converted into a symmetric matrix.
    N : int
        The size (number of rows/columns) of the output symmetric matrix.

    Returns
    -------
    np.ndarray
        The symmetric matrix of the specified size.
    
    References
    ----------
    - https://en.wikipedia.org/wiki/Vectorization_(mathematics)
    - K. G. Jinadasa, “Applications of the matrix operators vech and vec,” 
    Linear Algebra and its Applications, vol. 101, pp. 73–79, 1988, 
    doi: https://doi.org/10.1016/0024-3795(88)90143-7.
    - D. Nagakura, “Further results on the vecd operator and its applications,” 
    Communications in Statistics - Theory and Methods, vol. 49, no. 10, pp. 2321–2338, 2020, 
    doi: 10.1080/03610926.2019.1570265.
    """
    v = np.atleast_1d(v)
    A = np.zeros((N, N), dtype=v.dtype)
    tril_indices = np.tril_indices(N)
    A[tril_indices] = v
    A = A + A.T - np.diag(vecd(A))  # Make symmetric by adding transpose and removing double-counted diagonal
    return A

def vecd(A):
    """
    Diagonal vectorization of a matrix.

    Parameters
    ----------
    A : np.ndarray or list
        The matrix to be diagonal vectorized.

    Returns
    -------
    np.ndarray
        The diagonal vectorized form of the input matrix.
    
    References
    ----------
    - J. Brewer, "Kronecker products and matrix calculus in system theory," 
    in IEEE Transactions on Circuits and Systems, vol. 25, no. 9, pp. 772-781, 
    September 1978, doi: 10.1109/TCS.1978.1084534.
    """
    A = np.atleast_2d(A)
    return np.diag(A)

def unvecd(v):
    """
    Inverse of diagonal vectorization. Reshapes a vector back into a diagonal matrix.

    Parameters
    ----------
    v : np.ndarray or list
        The vector to be converted into a diagonal matrix.

    Returns
    -------
    np.ndarray
        The diagonal matrix of the specified size.
    
    References
    ----------
    - J. Brewer, "Kronecker products and matrix calculus in system theory," 
    in IEEE Transactions on Circuits and Systems, vol. 25, no. 9, pp. 772-781, 
    September 1978, doi: 10.1109/TCS.1978.1084534.
    """
    v = np.atleast_1d(v)
    return np.diag(v)

def vecb(A, block_size_row, block_size_col):
    """
    Block vectorization of a matrix.

    Parameters
    ----------
    A : np.ndarray or list
        The matrix to be block vectorized.
    block_size_row : list of int
        The height of each row block. len(block_size_row) is the number of row blocks.
    block_size_col : list of int
        The width of each column block. len(block_size_col) is the number of column blocks.
    
    Returns
    -------
    np.ndarray
        The block vectorized form of the input matrix.

    References
    ----------
    - R. H. Koning, H. Neudecker, and T. Wansbeek, 
    “Block Kronecker products and the vecb operator,” 
    Linear Algebra and its Applications, vol. 149, pp. 165–184, 1991, 
    doi: https://doi.org/10.1016/0024-3795(91)90332-Q.
    - D. S. Tracy, “Balanced partitioned matrices and their Kronecker products,” 
    Computational Statistics & Data Analysis, vol. 10, no. 3, pp. 315–323, 1990, 
    doi: https://doi.org/10.1016/0167-9473(90)90013-8.
    """
    A = np.atleast_2d(A)
    m = len(block_size_row)
    n = len(block_size_col)
    A_blocks = extract_blocks(A, block_size_row, block_size_col)
    return np.concatenate([vec(A_blocks[i][j]) for j in range(n) for i in range(m)])

def unvecb(v, block_size_row, block_size_col):
    """
    Inverse of block vectorization. Reshapes a vector back into a matrix with specified block structure.

    Parameters
    ----------
    v : np.ndarray or list
        The vector to be converted into a matrix.
    block_size_row : list of int
        The height of each row block. len(block_size_row) is the number of row blocks.
    block_size_col : list of int
        The width of each column block. len(block_size_col) is the number of column blocks.

    Returns
    -------
    np.ndarray
        The matrix of the specified block structure.

    References
    ----------
    - R. H. Koning, H. Neudecker, and T. Wansbeek, 
    “Block Kronecker products and the vecb operator,” 
    Linear Algebra and its Applications, vol. 149, pp. 165–184, 1991, 
    doi: https://doi.org/10.1016/0024-3795(91)90332-Q.
    - D. S. Tracy, “Balanced partitioned matrices and their Kronecker products,” 
    Computational Statistics & Data Analysis, vol. 10, no. 3, pp. 315–323, 1990, 
    doi: https://doi.org/10.1016/0167-9473(90)90013-8.
    """
    m = len(block_size_row)
    n = len(block_size_col)
    A_blocks = [[None for j in range(n)] for i in range(m)]
    
    idx = 0
    for j in range(n):
        for i in range(m):
            block_rows = block_size_row[i]
            block_cols = block_size_col[j]
            block_size = block_rows * block_cols
            A_blocks[i][j] = unvec(v[idx:idx+block_size], (block_rows, block_cols))
            idx += block_size

    return np.block([[A_blocks[i][j] for j in range(n)] for i in range(m)])

def vecdb(A, block_size):
    """
    Block diagonal vectorization of a matrix.

    Parameters
    ----------
    A : np.ndarray or list
        The matrix to be block diagonal vectorized.
    block_size: list of int
        The height/width of each block (square). len(block_size) is the number of blocks.
    
    Returns
    -------
    np.ndarray
        The block diagonal vectorized form of the input matrix.
    """
    A = np.atleast_2d(A)
    n = len(block_size)
    A_blocks = extract_blocks(A, block_size, block_size)
    return np.concatenate([vecd(A_blocks[i][j]) for j in range(n) for i in range(n)])

def unvecdb(v, block_size):
    """
    Inverse of block diagonal vectorization. Reshapes a vector back into a matrix with specified block diagonal structure.

    Parameters
    ----------
    v : np.ndarray or list
        The vector to be converted into a matrix.
    block_size: list of int
        The height/width of each block (square). len(block_size) is the number of blocks.

    Returns
    -------
    np.ndarray
        The matrix of the specified block diagonal structure.
    """
    n = len(block_size)
    A_blocks = [[None for j in range(n)] for i in range(n)]
    
    idx = 0
    for j in range(n):
        for i in range(n):
            N = block_size[i] # only diagonal elements, hence only N elements per block
            A_blocks[i][j] = unvecd(v[idx:idx+N])
            idx += N

    return np.block(A_blocks)

def kron(A, B):
    """
    Kronecker product of two matrices.

    Parameters
    ----------
    A : np.ndarray or list
        The first matrix.
    B : np.ndarray or list
        The second matrix.

    Returns
    -------
    np.ndarray
        The Kronecker product of the two input matrices.
    
    References
    ----------
    - https://en.wikipedia.org/wiki/Kronecker_product
    - K. B. Petersen and M. S. Pedersen, “The Matrix Cookbook.” 
    Technical University of Denmark, Nov. 2012. 
    [Online]. Available: http://www2.compute.dtu.dk/pubdb/pubs/3274-full.html
    - C. F. V. Loan, “The ubiquitous Kronecker product,” Journal of Computational and Applied Mathematics, 
    vol. 123, no. 1, pp. 85–100, 2000, doi: https://doi.org/10.1016/S0377-0427(00)00393-9.
    - J. Brewer, "Kronecker products and matrix calculus in system theory," 
    in IEEE Transactions on Circuits and Systems, vol. 25, no. 9, pp. 772-781, 
    September 1978, doi: 10.1109/TCS.1978.1084534.
    - H. Zhang and F. Ding, “On the Kronecker Products and Their Applications,” 
    Journal of Applied Mathematics, vol. 2013, no. 1, p. 296185, 2013, 
    doi: https://doi.org/10.1155/2013/296185.
    """
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    return np.kron(A, B)

def khatri(A, B):
    """
    Khatri-Rao column-wise product of two matrices.

    Parameters
    ----------
    A : np.ndarray or list
        The first matrix.
    B : np.ndarray or list
        The second matrix.

    Returns
    -------
    np.ndarray
        The Khatri-Rao product of the two input matrices.
    
    References
    ----------
    - https://en.wikipedia.org/wiki/Khatri%E2%80%93Rao_product
    - S. Liu and O. TRENKLER, “Hadamard, Khatri-Rao, Kronecker and other matrix products,” 
    International Journal of Information & Systems Sciences, vol. 4, Jan. 2008.
    - S. Liu, “Matrix results on the Khatri-Rao and Tracy-Singh products,” 
    Linear Algebra and its Applications, vol. 289, no. 1, pp. 267–277, 1999, 
    doi: https://doi.org/10.1016/S0024-3795(98)10209-4.
    """
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    if A.shape[1] != B.shape[1]:
        raise ValueError("The number of columns of A and B must be the same.")
    return np.vstack([np.kron(a, b) for a, b in zip(A.T, B.T)]).T

def hadamard(A, B):
    """
    Hadamard product of two matrices. Also known as the element-wise product.

    Parameters
    ----------
    A : np.ndarray or list
        The first matrix.
    B : np.ndarray or list
        The second matrix.

    Returns
    -------
    np.ndarray
        The Hadamard product of the two input matrices.
    
    References
    ----------
    - https://en.wikipedia.org/wiki/Hadamard_product_(matrices)
    - K. B. Petersen and M. S. Pedersen, “The Matrix Cookbook.”
    Technical University of Denmark, Nov. 2012.
    [Online]. Available: http://www2.compute.dtu.dk/pubdb/pubs/3274-full.html
    - S. Liu and O. TRENKLER, “Hadamard, Khatri-Rao, Kronecker and other matrix products,”
    International Journal of Information & Systems Sciences, vol. 4, Jan. 2008.
    """
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    if A.shape != B.shape:
        raise ValueError("The shapes of A and B must be the same.")
    return A*B

def block_kron(A, B, A_block_size_row, A_block_size_col, B_block_size_row, B_block_size_col):
    """
    Block Kronecker product of two matrices. Also known as the Tracy–Singh product.

    Parameters
    ----------
    A : np.ndarray
        The first matrix.
    B : np.ndarray
        The second matrix.
    A_block_size_row : list of int
        The height of each row block of A.
    A_block_size_col : list of int
        The width of each column block of A.
    B_block_size_row : list of int
        The height of each row block of B.
    B_block_size_col : list of int
        The width of each column block of B.

    Returns
    -------
    np.ndarray
        The block Kronecker product of the two input matrices.
    
    References
    ----------
    - R. H. Koning, H. Neudecker, and T. Wansbeek,
    “Block Kronecker products and the vecb operator,”
    Linear Algebra and its Applications, vol. 149, pp. 165–184, 1991,
    doi: https://doi.org/10.1016/0024-3795(91)90332-Q.
    - D. S. Tracy, “Balanced partitioned matrices and their Kronecker products,”
    Computational Statistics & Data Analysis, vol. 10, no. 3, pp. 315–323, 1990,
    doi: https://doi.org/10.1016/0167-9473(90)90013-8.
    """
    m = len(A_block_size_row)
    n = len(A_block_size_col)
    p = len(B_block_size_row)
    q = len(B_block_size_col)
    A_blocks = extract_blocks(A, A_block_size_row, A_block_size_col)
    B_blocks = extract_blocks(B, B_block_size_row, B_block_size_col)

    return np.block([[np.block([[kron(A_blocks[i][j], B_blocks[k][l]) for l in range(q)] for k in range(p)]) for j in range(n)] for i in range(m)])

def block_khatri(A, B, A_block_size_row, A_block_size_col, B_block_size_row, B_block_size_col):
    """
    Block Khatri-Rao column-wise product of two matrices.
    Only works if all blocks of A and B have same number of columns.

    Parameters
    ----------
    A : np.ndarray
        The first matrix.
    B : np.ndarray
        The second matrix.
    A_block_size_row : list of int
        The height of each row block of A.
    A_block_size_col : list of int
        The width of each column block of A.
    B_block_size_row : list of int
        The height of each row block of B.
    B_block_size_col : list of int
        The width of each column block of B.
    
    Returns
    -------
    np.ndarray
        The block Khatri-Rao product of the two input matrices.

    References
    ----------
    - D. S. Tracy, “Balanced partitioned matrices and their Kronecker products,”
    Computational Statistics & Data Analysis, vol. 10, no. 3, pp. 315–323, 1990,
    doi: https://doi.org/10.1016/0167-9473(90)90013-8.
    - S. Liu, “Matrix results on the Khatri-Rao and Tracy-Singh products,”
    Linear Algebra and its Applications, vol. 289, no. 1, pp. 267–277, 1999,
    doi: https://doi.org/10.1016/S0024-3795(98)10209-4.
    """
    m = len(A_block_size_row)
    n = len(A_block_size_col)
    p = len(B_block_size_row)
    q = len(B_block_size_col)

    if len(set(list(A_block_size_col) + list(B_block_size_col))) != 1:
        raise ValueError(f"All blocks must have same number of columns, but got A_col={list(A_block_size_col)}, B_col={list(B_block_size_col)}.")
    
    A_blocks = extract_blocks(A, A_block_size_row, A_block_size_col)
    B_blocks = extract_blocks(B, B_block_size_row, B_block_size_col)

    return np.block([[np.block([[khatri(A_blocks[i][j], B_blocks[k][l]) for l in range(q)] for k in range(p)]) for j in range(n)] for i in range(m)])

def commutation_matrix(m, n):
    """
    Commutation matrix of size m*n.

    Parameters
    ----------
    m : int
        Number of block rows.
    n : int
        Number of block columns.
    
    Returns
    -------
    np.ndarray
        The commutation matrix of size m*n.
    
    References
    ----------
    - https://en.wikipedia.org/wiki/Commutation_matrix
    - J. R. Magnus and H. Neudecker, “The Commutation Matrix: Some Properties and Applications,” 
    The Annals of Statistics, vol. 7, no. 2, pp. 381–394, 1979, doi: 10.1214/aos/1176344621.
    """
    idx = vec(np.arange(m*n).reshape(n,m))
    return np.eye(m*n, dtype=int)[idx]

if __name__ == "__main__":
    # Example: block Kronecker (Tracy–Singh) manual vs function
    A = np.array([[1, 2],
                  [3, 4],
                  [5, 6],
                  [7, 8]])  # 4x2

    B = np.array([[1, 0, 5],
                  [2, 1, 6],
                  [3, 4, 7]])  # 3x3

    # A blocked as 2x1, with first block 2x2
    A_block_size_row = [2, 2]
    A_block_size_col = [2]

    # B blocked as 2x2 (top-left), then remaining 1 row / 1 column
    B_block_size_row = [2, 1]
    B_block_size_col = [2, 1]

    # Function output
    result_code = block_kron(A, B, A_block_size_row, A_block_size_col, B_block_size_row, B_block_size_col)

    # Manual output (explicit block construction) matching the same block partitions
    A11 = A[0:2, 0:2]
    A21 = A[2:4, 0:2]

    B11 = B[0:2, 0:2]
    B12 = B[0:2, 2:3]
    B21 = B[2:3, 0:2]
    B22 = B[2:3, 2:3]

    result_manual = np.block([[np.kron(A11, B11), np.kron(A11, B12)],
                              [np.kron(A11, B21), np.kron(A11, B22)],
                              [np.kron(A21, B11), np.kron(A21, B12)],
                              [np.kron(A21, B21), np.kron(A21, B22)]])

    sum_error = np.sum(np.abs(result_manual - result_code))

    print("Sum error:   ", sum_error)