# Matrix Vectorization and Kronecker Products

`kronutils.py` is a pure-NumPy utility library for matrix vectorization operators and Kronecker products. All functions accept NumPy arrays or nested Python lists.

| Function | Description |
|---|---|
| `vec(A)` | Column-wise vectorization |
| `unvec(v, shape)` | Inverse of column-wise vectorization |
| `vech(A)` | Half-vectorization (lower triangle, square matrices only) |
| `unvech(v, N)` | Inverse of half-vectorization (reconstructs symmetric matrix) |
| `vecd(A)` | Diagonal vectorization |
| `unvecd(v)` | Inverse of diagonal vectorization |
| `vecb(A, block_size_row, block_size_col)` | Block vectorization |
| `unvecb(v, block_size_row, block_size_col)` | Inverse of block vectorization |
| `vecdb(A, block_size)` | Block diagonal vectorization |
| `unvecdb(v, block_size)` | Inverse of block diagonal vectorization |
| `extract_blocks(A, block_size_row, block_size_col)` | Extract all blocks into a 2D list |
| `extract_diag_blocks(A, block_size_row, block_size_col)` | Extract diagonal blocks into a list |
| `kron(A, B)` | Kronecker product |
| `khatri(A, B)` | Khatri-Rao column-wise product |
| `hadamard(A, B)` | Hadamard (element-wise) product |
| `block_kron(A, B, ...)` | Block Kronecker product (Tracy–Singh product) |
| `block_khatri(A, B, ...)` | Block Khatri-Rao product |
| `block_diag(*matrices)` | Block diagonal matrix (direct sum) |
| `commutation_matrix(m, n)` | Commutation matrix of size *mn × mn* |

## Requirements

The file `kronutils.py` should be imported into your script (see example below). Best to have the file in the same location as your script, otherwise you need to specify the directory.

You need [NumPy](https://github.com/numpy/numpy) installed.
```powershell
python -m pip install numpy -U
```

## Example Usage

```python
import numpy as np
from kronutils import (vec, unvec, vech, unvech, vecd, unvecd,
                       vecb, unvecb, vecdb, unvecdb,
                       extract_blocks, extract_diag_blocks,
                       kron, khatri, hadamard, block_kron, block_khatri,
                       block_diag, commutation_matrix)

A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

# Column-wise vectorization
print(vec(A))           # [1 3 2 4]
print(unvec(vec(A), (2, 2)))  # recovers A

# Half-vectorization (lower triangle)
print(vech(A))          # [1 3 4]
print(unvech(vech(A), 2))     # recovers A (symmetric)

# Diagonal vectorization
print(vecd(A))          # [1 4]
print(unvecd(vecd(A)))  # [[1 0], [0 4]]

# Kronecker product
print(kron(A, B))
# [[ 5  6 10 12]
#  [ 7  8 14 16]
#  [15 18 20 24]
#  [21 24 28 32]]

# Khatri-Rao product
print(khatri(A, B))
# [[ 5 12]
#  [ 7 16]
#  [15 24]
#  [21 32]]

# Hadamard product
print(hadamard(A, B))
# [[ 5 12]
#  [21 32]]

# Commutation matrix: vec(A) = K @ vec(A.T)
K = commutation_matrix(2, 2)
print(K @ vec(A.T))     # [1 3 2 4]  ==  vec(A)
```

**Block vectorization:**

```python
C = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

block_size_row = [2, 1]
block_size_col = [2, 1]

v = vecb(C, block_size_row, block_size_col)
print(unvecb(v, block_size_row, block_size_col))  # recovers C

# Extract blocks
blocks = extract_blocks(C, block_size_row, block_size_col)
# blocks[0][0] == C[0:2, 0:2], blocks[0][1] == C[0:2, 2:3], etc.

diag_blocks = extract_diag_blocks(C, block_size_row, block_size_col)
# diag_blocks[0] == C[0:2, 0:2],  diag_blocks[1] == C[2:3, 2:3]
```

**Block Kronecker (Tracy–Singh) product:**

```python
from kronutils import block_kron

A = np.array([[1, 2],
              [3, 4],
              [5, 6],
              [7, 8]])   # 4×2, blocked as [2,2] × [2]

B = np.array([[1, 0, 5],
              [2, 1, 6],
              [3, 4, 7]])  # 3×3, blocked as [2,1] × [2,1]

result = block_kron(A, B,
                    A_block_size_row=[2, 2], A_block_size_col=[2],
                    B_block_size_row=[2, 1], B_block_size_col=[2, 1])
print(result)
```

## References

- J. Brewer, "Kronecker products and matrix calculus in system theory," *IEEE Transactions on Circuits and Systems*, vol. 25, no. 9, pp. 772–781, Sep. 1978, doi: [10.1109/TCS.1978.1084534](https://doi.org/10.1109/TCS.1978.1084534).

- K. G. Jinadasa, "Applications of the matrix operators vech and vec," *Linear Algebra and its Applications*, vol. 101, pp. 73–79, 1988, doi: [10.1016/0024-3795(88)90143-7](https://doi.org/10.1016/0024-3795(88)90143-7).

- R. H. Koning, H. Neudecker, and T. Wansbeek, "Block Kronecker products and the vecb operator," *Linear Algebra and its Applications*, vol. 149, pp. 165–184, 1991, doi: [10.1016/0024-3795(91)90332-Q](https://doi.org/10.1016/0024-3795(91)90332-Q).

- S. Liu, "Matrix results on the Khatri-Rao and Tracy-Singh products," *Linear Algebra and its Applications*, vol. 289, no. 1, pp. 267–277, 1999, doi: [10.1016/S0024-3795(98)10209-4](https://doi.org/10.1016/S0024-3795(98)10209-4).

- C. F. Van Loan, "The ubiquitous Kronecker product," *Journal of Computational and Applied Mathematics*, vol. 123, no. 1, pp. 85–100, 2000, doi: [10.1016/S0377-0427(00)00393-9](https://doi.org/10.1016/S0377-0427(00)00393-9).

- K. B. Petersen and M. S. Pedersen, "The Matrix Cookbook," Technical University of Denmark, Nov. 2012. [Online]. Available: http://www2.compute.dtu.dk/pubdb/pubs/3274-full.html

- D. S. Tracy, "Balanced partitioned matrices and their Kronecker products," *Computational Statistics & Data Analysis*, vol. 10, no. 3, pp. 315–323, 1990, doi: [10.1016/0167-9473(90)90013-8](https://doi.org/10.1016/0167-9473(90)90013-8).

- J. R. Magnus and H. Neudecker, "The Commutation Matrix: Some Properties and Applications," *The Annals of Statistics*, vol. 7, no. 2, pp. 381–394, 1979, doi: [10.1214/aos/1176344621](https://doi.org/10.1214/aos/1176344621).

- S. Liu and G. Trenkler, "Hadamard, Khatri-Rao, Kronecker and other matrix products," *International Journal of Information & Systems Sciences*, vol. 4, Jan. 2008.

- D. Nagakura, "Further results on the vecd operator and its applications," *Communications in Statistics – Theory and Methods*, vol. 49, no. 10, pp. 2321–2338, 2020, doi: [10.1080/03610926.2019.1570265](https://doi.org/10.1080/03610926.2019.1570265).

- H. Zhang and F. Ding, "On the Kronecker Products and Their Applications," *Journal of Applied Mathematics*, vol. 2013, p. 296185, 2013, doi: [10.1155/2013/296185](https://doi.org/10.1155/2013/296185).

- Wikipedia, "Direct sum," <https://en.wikipedia.org/wiki/Direct_sum>

- Wikipedia, "Vectorization (mathematics)," <https://en.wikipedia.org/wiki/Vectorization_(mathematics)>

- Wikipedia, "Kronecker product," <https://en.wikipedia.org/wiki/Kronecker_product>

- Wikipedia, "Khatri–Rao product," <https://en.wikipedia.org/wiki/Khatri%E2%80%93Rao_product>

- Wikipedia, "Hadamard product (matrices)," <https://en.wikipedia.org/wiki/Hadamard_product_(matrices)>

- Wikipedia, "Commutation matrix," <https://en.wikipedia.org/wiki/Commutation_matrix>

## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)