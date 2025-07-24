import math
import numpy as np
from matplotlib import pyplot
import os

# %%
# 1. Random Data Generator

# a.

def gaussian_data_generator():
    m = float(input('Please enter the desired mean value, m = '))
    s = float(input('Please enter the desired variance(>0) value, s = '))
    print(f'Data point source function: N({m}, {s})')
    print("\n\n")
    data_col = []
    n = 1
    mean, var = 100, 100
    while abs(m - mean) > 0.01 or abs(s - var) > 0.01:
        # Box–Muller method
        u1 =  np.random.uniform(0, 1)
        u2 =  np.random.uniform(0, 1)
        z = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
        data = m + math.sqrt(s) *z
        print("Add data point:", data)
        data_col.append(data)
        sum = 0
        for i in data_col:
            sum += i 
        mean = sum / n
        sum = 0
        for j in data_col:
            sum += (j - mean)**2
        if n == 1:
            var = 0
        else:
            var = sum / (n-1)
        print(f'Mean = {mean} Variance = {var}')
        n += 1
    print(f'when n = {n}, the (mean, variance) is converge to ({m}, {s})')  

# %%

# b.

def polynomial_basis(x, n):
    return np.array([x**i for i in range(n)])

def polynomial_basis_linear_model_data_generator(n, a, w):
    x = np.random.uniform(-1, 1)
    phi = polynomial_basis(x, n)
    sum = 0
    for i in range(n):
        sum += w[i] * phi[i] 
    # Box–Muller method
    u1 =  np.random.uniform(0, 1)
    u2 =  np.random.uniform(0, 1)
    z = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
    y = sum + math.sqrt(a) * z

    return (x, y)

# %% 2. N(3.0, 5.0)

gaussian_data_generator()


# %% 4.

def zeros(shape, dtype=float):

    if isinstance(shape, tuple):
        rows, cols = shape
        return [[dtype(0) for _ in range(cols)] for _ in range(rows)]
    else:
        return [[dtype(0) for _ in range(shape)]]

def eye(n, dtype=float):

    matrix = [[dtype(0) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        matrix[i][i] = dtype(1)
    return matrix

def lu_decomposition(A):
    n1 = len(A)
    U = [row[:] for row in A]  
    L = eye(n1)
    
    for j in range(n1 - 1):
        if U[j][j] == 0:  
            non_zero_row = [i for i in range(j+1, n1) if U[i][j] != 0]
            if len(non_zero_row) == 0:
                raise ValueError("矩陣不可逆，無法進行 LU 分解")
            U[j], U[non_zero_row[0]] = U[non_zero_row[0]], U[j]
            L[j], L[non_zero_row[0]] = L[non_zero_row[0]], L[j]
        
        for i in range(j + 1, n1):
            factor = U[i][j] / U[j][j]
            U[i] = [U[i][k] - factor * U[j][k] for k in range(n1)]
            L[i][j] = factor
    return L, U

def inverse(A):
    n1 = len(A)
    L, U = lu_decomposition(A)
    I = eye(n1)
    for i in range(1, n1):
       for j in range(i):
           sum = L[i][j]
           for k in range(j+1, i):
               sum += L[i][k] * I[k][j]
           I[i][j] = -sum
    A_inv = [[0 for _ in range(n1)] for _ in range(n1)]
    for i in range(n1-1, -1, -1):
        for j in range(n1):
           sum = 0
           for k in range(i+1, n1):
               sum += U[i][k] * A_inv[k][j]
           A_inv[i][j] = (I[i][j] - sum)/ U[i][i]
    return np.array(A_inv)


def matrix_multiply(A, B):
    """Matrix multiplication"""
    if len(A[0]) != len(B):
        raise ValueError(f"矩陣維度不匹配：A的列數 ({len(A[0])}) 不等於 B的行數 ({len(B)})")
    
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(len(B)))
    return result

def transpose(matrix):
    """Matrix transpose"""
    if not matrix:
        return []
    return [[matrix[j][i] for j in range(len(matrix))] 
            for i in range(len(matrix[0]))]

def matrix_scalar_multiply(matrix, scalar):
    """Multiply matrix by scalar"""
    return [[scalar * x for x in row] for row in matrix]

# %%

def bayesian_linear_regression():
    result = open("result.txt", 'w')
    b = float(input("b = "))
    n = int(input("n = "))
    a = float(input("a = "))
    a = 1/a
    w = zeros((n, 1), dtype=float)
    for i in range(n):
        w[i][0] = float(input(f"w[{i}] = "))

    points = []
    mean = zeros((n, 1), dtype=float)
    var = [[x * (1/b) for x in row] for row in eye(n)]
    predic_mean = 0.
    predic_var = 0.
    err = 100
    count = 0

    f, axs = pyplot.subplots(2, 2, figsize=(10, 10))
    
    # Ground truth plot
    axs[0][0].set_title("Ground truth")
    axs[0][0].set_xlim(-2, 2)
    axs[0][0].set_ylim(-10, 25)
    x_value = np.linspace(-2, 2, 1000)  
    y_pred = [0] * len(x_value)
    for i in range(len(w)):
        for j in range(len(x_value)):
            y_pred[j] += w[i][0] * (x_value[j] ** i)
    axs[0][0].plot(x_value, y_pred, color='black')
    
    # Add confidence bounds
    UB = [y + (1/a) for y in y_pred]
    LB = [y - (1/a) for y in y_pred]
    axs[0][0].plot(x_value, UB, color='r')
    axs[0][0].plot(x_value, LB, color='r')

    while err > 1e-6:
        new_data = polynomial_basis_linear_model_data_generator(n, a, [w[0] for w in w])
        points.append(new_data)
        count += 1
        result.write(f"Add data point {new_data}\n\n")
    
        phi = zeros((1, n))
        for i in range(n):
            phi[0][i] = new_data[0] ** i
        y = new_data[1]
    
        S = inverse(var)
        
        phi_trans = transpose(phi)
        
        phit_phi = matrix_multiply(phi_trans, phi)
        phit_phi_scaled = matrix_scalar_multiply(phit_phi, 1/a)
        C = [[phit_phi_scaled[i][j] + S[i][j] for j in range(n)] for i in range(n)]
        
        a_y_phit = matrix_scalar_multiply(phi_trans, (1/a) * y)
        Smean = matrix_multiply(S, mean)
        sum_term = [[a_y_phit[i][0] + Smean[i][0]] for i in range(n)]
        
        mu = matrix_multiply(inverse(C), sum_term)
    
        new_mean = matrix_multiply(phi, mu)[0][0]
        inv_C = inverse(C)
        phi_invC = matrix_multiply(phi, inv_C)
        new_var = (1/a) + matrix_multiply(phi_invC, phi_trans)[0][0]
        
        err = abs(new_var - predic_var)
        predic_mean = new_mean
        predic_var = new_var
    
        mean = mu
        var = inv_C

        # Plot after 10 incomes
        if count == 10:
            axs[1][0].set_title("After 10 incomes")
            axs[1][0].set_xlim(-2, 2)
            axs[1][0].set_ylim(-10, 25)
            
            # Plot ground truth
            y_pred = [0] * len(x_value)
            for i in range(len(w)):
                for j in range(len(x_value)):
                    y_pred[j] += w[i][0] * (x_value[j] ** i)
            axs[1][0].plot(x_value, y_pred, color='black')
            
            # Plot data points
            x_points = [point[0] for point in points]
            y_points = [point[1] for point in points]
            axs[1][0].scatter(x_points, y_points)
            
            # Plot confidence bounds
            UB = [0] * len(x_value)
            LB = [0] * len(x_value)
            for i in range(len(x_value)):
                X = zeros((1, len(w)))
                for j in range(len(w)):
                    X[0][j] = x_value[i] ** j
                X_trans = transpose(X)
                variance_term = matrix_multiply(matrix_multiply(X, var), X_trans)[0][0]
                UB[i] = y_pred[i] + (1/a) + variance_term
                LB[i] = y_pred[i] - (1/a) - variance_term
            axs[1][0].plot(x_value, UB, color='r')
            axs[1][0].plot(x_value, LB, color='r')
            
        # Plot after 50 incomes
        elif count == 50:
            axs[1][1].set_title("After 50 incomes")
            axs[1][1].set_xlim(-2, 2)
            axs[1][1].set_ylim(-10, 25)
            
            # Plot ground truth
            y_pred = [0] * len(x_value)
            for i in range(len(w)):
                for j in range(len(x_value)):
                    y_pred[j] += w[i][0] * (x_value[j] ** i)
            axs[1][1].plot(x_value, y_pred, color='black')
            
            # Plot data points
            x_points = [point[0] for point in points]
            y_points = [point[1] for point in points]
            axs[1][1].scatter(x_points, y_points)
            
            # Plot confidence bounds
            UB = [0] * len(x_value)
            LB = [0] * len(x_value)
            for i in range(len(x_value)):
                X = zeros((1, len(w)))
                for j in range(len(w)):
                    X[0][j] = x_value[i] ** j
                X_trans = transpose(X)
                variance_term = matrix_multiply(matrix_multiply(X, var), X_trans)[0][0]
                UB[i] = y_pred[i] + (1/a) + variance_term
                LB[i] = y_pred[i] - (1/a) - variance_term
            axs[1][1].plot(x_value, UB, color='r')
            axs[1][1].plot(x_value, LB, color='r')
        
        result.write("Posterior mean:\n")
        for i in range(n):
            result.write(f" {mean[i][0]}\n")
        result.write("\nPosterior variance:\n")
        for i in range(n):
           for j in range(n):
                result.write(f" {var[i][j]}, ")
           result.write("\n")

        result.write(f"\nPredictive distribution ~ N({predic_mean}, {predic_var})\n")
        result.write("--------------------------------------------------\n")
    # Plot final predict result
    axs[0][1].set_title("Predict result")
    axs[0][1].set_xlim(-2, 2)
    axs[0][1].set_ylim(-10, 25)
    
    # Plot ground truth
    y_pred = [0] * len(x_value)
    for i in range(len(mean)):
        for j in range(len(x_value)):
            y_pred[j] += mean[i][0] * (x_value[j] ** i)
    axs[0][1].plot(x_value, y_pred, color='black')
    
    # Plot data points
    x_points = [point[0] for point in points]
    y_points = [point[1] for point in points]
    axs[0][1].scatter(x_points, y_points)
    
    # Plot confidence bounds
    UB = [0] * len(x_value)
    LB = [0] * len(x_value)
    for i in range(len(x_value)):
        X = zeros((1, len(mean)))
        for j in range(len(mean)):
            X[0][j] = x_value[i] ** j
        X_trans = transpose(X)
        variance_term = matrix_multiply(matrix_multiply(X, var), X_trans)[0][0]
        UB[i] = y_pred[i] + (1/a) + variance_term
        LB[i] = y_pred[i] - (1/a) - variance_term
    axs[0][1].plot(x_value, UB, color='r')
    axs[0][1].plot(x_value, LB, color='r')

    pyplot.show()
    os.startfile('result.txt')
# %% b = 1, n = 4, a = 1, w = [1, 2, 3, 4] 

bayesian_linear_regression()

# %%  b = 100, n = 4, a = 1, w = [1, 2, 3, 4]

bayesian_linear_regression()

# %%  b = 1, n = 3, a = 3, w = [1, 2, 3] 

bayesian_linear_regression()