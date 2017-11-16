""" 矩阵例子 """
import numpy as np

matrix_a = np.mat('4 3 2; 2 1 4; 1 2 3')
matrix_b = np.mat('1 2 3; 2 3 4; 3 4 1')

print(matrix_a)
[[4 3 2]
 [2 1 4]
 [1 2 3]]
print(matrix_b)
[[1 2 3]
 [2 3 4]
 [3 4 1]]
print(matrix_a * matrix_b)
[[16 25 26]
 [16 23 14]
 [14 20 14]]

print(matrix_a - matrix_b)

print(matrix_a + matrix_b)
