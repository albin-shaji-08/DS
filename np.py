# ===============================
# 1. Differentiate between Python list and NumPy array (example)
# ===============================
import numpy as np

# Python list
py_list = [1, 2, 3, 4, 5]
# NumPy array
np_array = np.array([1, 2, 3, 4, 5])

# ===============================
# 2. Create 1D, 2D, and 3D arrays using np.array(), np.zeros(), np.ones(), np.arange()
# ===============================
arr_1d = np.array([1, 2, 3, 4, 5])
arr_2d = np.array([[1, 2], [3, 4]])
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

zeros_arr = np.zeros((2,3))
ones_arr = np.ones((3,2))
arange_arr = np.arange(0,10,2)

# ===============================
# 3. Create a 5x5 identity matrix
# ===============================
identity = np.eye(5)

# ===============================
# 4. Generate random numbers
# ===============================
rand_integers = np.random.randint(1,101,10)
rand_floats = np.random.rand(3,3)

# ===============================
# 5. Arithmetic on two arrays
# ===============================
a = np.array([10,20,30])
b = np.array([1,2,3])

add = a + b
sub = a - b
mul = a * b
div = a / b

# ===============================
# 6. Broadcasting in NumPy
# ===============================
arr_2d = np.array([[1,2,3],[4,5,6]])
arr_1d = np.array([10,20,30])
broadcasted = arr_2d + arr_1d  # Adds 1D array to each row

# ===============================
# 7. Compute statistical measures
# ===============================
scores = np.array([56, 67, 45, 88, 72, 90, 61, 76, 84, 69])
mean = np.mean(scores)
median = np.median(scores)
std_dev = np.std(scores)
variance = np.var(scores)

# ===============================
# 8. Slice a 2D array to extract a submatrix
# ===============================
arr4x4 = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
submatrix_2x2 = arr4x4[1:3, 1:3]

# ===============================
# 9. Reshape a 1D array into 2D forms
# ===============================
arr_1d_12 = np.arange(1,13)
arr_3x4 = arr_1d_12.reshape(3,4)
arr_2x6 = arr_1d_12.reshape(2,6)

# ===============================
# 10. Flatten a multi-dimensional array
# ===============================
arr_2d_example = np.array([[1,2,3],[4,5,6]])
flat_arr = arr_2d_example.flatten()
ravel_arr = arr_2d_example.ravel()

# ===============================
# 11. Sort arrays
# ===============================
arr1d_sort = np.array([50, 20, 10, 40, 30])
sorted_1d = np.sort(arr1d_sort)

arr2d_sort = np.array([[3,2,1],[6,5,4]])
sorted_rows = np.sort(arr2d_sort, axis=1)
sorted_cols = np.sort(arr2d_sort, axis=0)

argsorted_indices = np.argsort(arr1d_sort)

# ===============================
# 12. Conditional selection with np.where()
# ===============================
arr_cond = np.array([5,12,7,15,3,20])
cond_result = np.where(arr_cond > 10, 1, 0)

# ===============================
# 13. Matrix multiplication
# ===============================
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
matmul_dot = np.dot(A,B)
matmul_at = A @ B

# ===============================
# 14. Inverse and determinant of a matrix
# ===============================
C = np.array([[2,3],[1,4]])
det_C = np.linalg.det(C)
inv_C = np.linalg.inv(C)
identity_check = np.dot(C, inv_C)

# ===============================
# 15. Solve system of equations
# ===============================
# 2x + y = 8, 3x + 2y = 13
coeff = np.array([[2,1],[3,2]])
const = np.array([8,13])
solution = np.linalg.solve(coeff, const)

# ===============================
# 16. Read CSV and compute average of a column
# ===============================
 data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)
 avg_col = np.mean(data[:,1])  # Example: Salary colum

# ===============================
# 17. Simulate student scores
# ===============================
marks = np.random.randint(0,101,100)
avg_marks = np.mean(marks)
highest_marks = np.max(marks)
lowest_marks = np.min(marks)
pass_percentage = (np.sum(marks >= 50)/len(marks))*100

# ===============================
# Print some results as example (optional)
# ===============================
print("Submatrix 2x2:\n", submatrix_2x2)
print("Flattened array:", flat_arr)
print("Ravel array:", ravel_arr)
print("Broadcasted array:\n", broadcasted)
print("Mean:", mean, "Median:", median, "Std Dev:", std_dev, "Variance:", variance)
print("Matrix multiplication @ operator:\n", matmul_at)
print("Inverse of C:\n", inv_C)
print("Solution of equations [x, y]:", solution)
print("Pass percentage:", pass_percentage, "%")
