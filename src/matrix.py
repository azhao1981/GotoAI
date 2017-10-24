""" 矩阵例子 """
import numpy as np

matrix_a = np.mat('4 3; 2 1')
matrix_b = np.mat('1 2; 3 4')

print(matrix_a)
# [[4 3]
#  [2 1]]
print(matrix_b)
# [[1 2]
#  [3 4]]
print(matrix_a*matrix_b)
# [[13 20]
#  [ 5  8]]

print(matrix_a - matrix_b)

print(matrix_a * matrix_b)
