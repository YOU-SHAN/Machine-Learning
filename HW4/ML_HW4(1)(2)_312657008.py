import math
import numpy as np
import matplotlib.pyplot as plt
import os


# %% 1.
def zeros(shape, dtype=float):

    if isinstance(shape, tuple):
        rows, cols = shape
        return [[dtype(0) for _ in range(cols)] for _ in range(rows)]
    else:
        return [[dtype(0) for _ in range(shape)]]
    
def ones(shape, dtype=float):

    if isinstance(shape, tuple):
        rows, cols = shape
        return [[dtype(1) for _ in range(cols)] for _ in range(rows)]
    else:
        return [[dtype(1) for _ in range(shape)]] 

def eye(n, dtype=float):

    matrix = [[dtype(0) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        matrix[i][i] = dtype(1)
    return matrix

def transpose(matrix):
    """Matrix transpose"""
    return [[matrix[j][i] for j in range(len(matrix))] 
            for i in range(len(matrix[0]))]

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
    
# %%
class Logistic_regression:
    def input_value(self):
        self.n = int(input("Input the number of data points: "))
        self.mx1 = float(input("Input mx1: "))
        self.vx1 = float(input("Input vx1 (>0): "))
        self.my1 = float(input("Input my1: "))
        self.vy1 = float(input("Input vy1 (>0): "))
        self.mx2 = float(input("Input mx2: "))
        self.vx2 = float(input("Input vx2 (>0): "))
        self.my2 = float(input("Input my2: "))
        self.vy2 = float(input("Input vy2 (>0): "))

    def generate_data(self, mean_x, var_x, mean_y, var_y):
        x, y = [], []
        for _ in range(self.n):
            # 使用 Box–Muller 方法生成常態分布數據
            u1, u2 = np.random.uniform(0, 1), np.random.uniform(0, 1)
            z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            x.append(mean_x + math.sqrt(var_x) * z)
            u1, u2 = np.random.uniform(0, 1), np.random.uniform(0, 1)
            z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            y.append(mean_y + math.sqrt(var_y) * z)
        return np.array(x), np.array(y)

    def D1(self):
        return self.generate_data(self.mx1, self.vx1, self.my1, self.vy1)
    
    def D2(self):
        return self.generate_data(self.mx2, self.vx2, self.my2, self.vy2)
    
    def parameter(self):
        w = [[0.] for _ in range(3)]
        x1, y1 = Logistic_regression.D1(self)
        x2, y2 = Logistic_regression.D2(self)
        X = ones(( 2* self.n, 3), dtype= float)
        for i in range(self.n):
            for j in range(3):
                if j == 1:
                    X[i][j] = x1[i]
                elif j == 2:
                    X[i][j] = y1[i]
        for i in range(self.n):
            for j in range(3):
                if j == 1:
                    X[i+self.n][j] = x2[i]
                elif j == 2:
                    X[i+self.n][j] = y2[i]
        Y = np.array([int(1) for _ in range(2*self.n)])
        for i in range(self.n):
            Y[i] = int(0)
        return w, X, Y
    def norm(w):
        diff = 0
        for i in w:
            diff += i[0] ** 2
        return diff        
    def Ground_truth(self, ax):
        w, X, Y = Logistic_regression.parameter(self)
        for i in range(2 * self.n):
            if Y[i] == 0:
                ax.scatter(X[i][1], X[i][2], c='red', s=24)
            else:
                ax.scatter(X[i][1], X[i][2], c='blue', s=24)
        ax.set_title('Ground Truth')

        plt.show()
    def Gradient_descent(self, ax, learning_rate=0.01):
        w, X, Y = Logistic_regression.parameter(self)
        diff = 1.
        temp = [[0.] for _ in range(2 * self.n)]
        while diff > 1e-4:
            for i in range(2 * self.n):
                a = 0
                for j in range(3):
                    a += X[i][j] * w[j][0]
                temp[i][0] = Y[i] - (1/(1+math.exp(-a)))
            X_t = transpose(X)
            dw = [[0.] for _ in range(3)]
            for k in range(3):
                c = 0
                for l in range(2 * self.n):
                    c += X_t[k][l] * temp[l][0]
                dw[k][0] = c 
            #更新w
            w_new = [[0.] for _ in range(3)]
            for p in range(3):
                w_new[p][0] = w[p][0] + learning_rate*dw[p][0]
            diff = Logistic_regression.norm(dw)
            w = w_new
        print("\n")
        print(f'Gradient descent:\n')
        print(f'w:\n')
        for element in w: 
            print(f'{element}\n')
        #預測
        prob = [[0.] for _ in range(2 * self.n)]
        for i in range(2 * self.n):
           a = 0
           for j in range(3):
                a += X[i][j] * w[j][0]
           prob[i][0] = (1/(1+math.exp(-a)))
        pred = [[int(0)] for _ in range(2 * self.n)]   
        for i in range(2 * self.n):
            if prob[i][0] > 0.5:
                pred[i][0] = 1
        TP, FN, FP, TN = 0, 0, 0, 0
        
        for i in range(self.n):
            if pred[i][0] == 0 :
                TP += 1
            if pred[i][0] == 1:
                FN += 1
            if pred[i+self.n][0] == 0:
                FP += 1
            if pred[i+self.n][0] == 1:
                TN += 1
        print('Confusion Matrix:\n')
        print('               Predict cluster 0  Predict cluster 1\n')
        print(f'Is cluster 0       {TP}              {FN}\n')
        print(f'Is cluster 1       {FP}              {TN}\n')
        sensitivity = TP / (TP + FN)
        specificity = TN / (FP + TN) 
        print(f'Sensitivity (Successfully predict cluster 0): {sensitivity}\n')
        print(f'Specificity (Successfully predict cluster 1): {specificity}\n')
        for i in range(2 * self.n):
            if pred[i][0] == 0:
                ax.scatter(X[i][1], X[i][2], c='red', s=24)
            else:
                ax.scatter(X[i][1], X[i][2], c='blue', s=24)
        ax.set_title('Gradient Descent')
        
    def Newton_method(self, ax, learning_rate=0.01):
            w, X, Y = Logistic_regression.parameter(self)
            diff = 1.
            temp = [[0.] for _ in range(2 * self.n)]
            while diff > 1e-4:
                
                for i in range(2 * self.n):
                    a = 0
                    for j in range(3):
                        a += X[i][j] * w[j][0]
                    temp[i][0] = Y[i] - (1/(1+math.exp(-a)))
                X_t = transpose(X)
                dw = [[0.] for _ in range(3)]
                for k in range(3):
                    c = 0
                    for l in range(2 * self.n):
                        c += X_t[k][l] * temp[l][0]
                    dw[k][0] = c
                D = zeros((2 * self.n, 2 * self.n), dtype=float)
                for i in range(2 * self.n):
                    a = 0
                    for k in range(3):  
                        a += X[i][k] * w[k][0]
                    D[i][i] = math.exp(-a) / ((1 + math.exp(-a)) ** 2)
                try:
                    H = matrix_multiply(matrix_multiply(X_t, D), X)    
                    inv_H = inverse(H)
                    H_inv_dw = [[0.] for _ in range(3)]
                    for k in range(3):
                        c = 0
                        for l in range(3):
                            c += H_inv_dw[k][l] * dw[l][0]
                        H_inv_dw[k][0] = c 
                    #更新w
                    w_new = [[0.] for _ in range(3)]
                    for p in range(3):
                        w_new[p][0] = w[p][0] + learning_rate*H_inv_dw[p][0]
                    diff = Logistic_regression.norm(H_inv_dw)
                except:
                    #更新w
                    w_new = [[0.] for _ in range(3)]
                    for p in range(3):
                        w_new[p][0] = w[p][0] + learning_rate*dw[p][0]
                    diff = Logistic_regression.norm(dw)
                w = w_new
            print("---------------------------------------\n")
            print(f"Newton's method: \n")
            print(f'w:\n')
            for element in w: 
                print(f'{element}\n')
            #預測
            prob = [[0.] for _ in range(2 * self.n)]
            for i in range(2 * self.n):
               a = 0
               for j in range(3):
                    a += X[i][j] * w[j][0]
               prob[i][0] = (1/(1+math.exp(-a)))
            pred = [[int(0)] for _ in range(2 * self.n)]   
            for i in range(2 * self.n):
                if prob[i][0] > 0.5:
                    pred[i][0] = 1
            TP, FN, FP, TN = 0, 0, 0, 0
            
            for i in range(self.n):
                if pred[i][0] == 0 :
                    TP += 1
                if pred[i][0] == 1:
                    FN += 1
                if pred[i+self.n][0] == 0:
                    FP += 1
                if pred[i+self.n][0] == 1:
                    TN += 1
            print('Confusion Matrix:\n')
            print('               Predict cluster 0  Predict cluster 1\n')
            print(f'Is cluster 0       {TP}              {FN}\n')
            print(f'Is cluster 1       {FP}              {TN}\n')
            sensitivity = TP / (TP + FN)
            specificity = TN / (FP + TN) 
            print(f'Sensitivity (Successfully predict cluster 0): {sensitivity}\n')
            print(f'Specificity (Successfully predict cluster 1): {specificity}\n')
            for i in range(2 * self.n):
                if pred[i][0] == 0:
                    ax.scatter(X[i][1], X[i][2], c='red', s=24)
                else:
                    ax.scatter(X[i][1], X[i][2], c='blue', s=24)
            ax.set_title("Newton's Method")
# %%
if __name__ == "__main__":
    model = Logistic_regression()  
    model.input_value()  
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    model.Ground_truth(axs[0])
    model.Gradient_descent(axs[1])
    model.Newton_method(axs[2])
    
    plt.tight_layout()
    plt.show()

# %% 2. 


def load():
    label_file = open(r"C:\Users\yoush\Desktop\機器學習\HW4\train-labels.idx1-ubyte__", "rb")
    image_file = open(r"C:\Users\yoush\Desktop\機器學習\HW4\train-images.idx3-ubyte__", "rb")

    label_file.read(8)
    image_file.read(4)
    number = int.from_bytes(image_file.read(4), byteorder='big')
    row = int.from_bytes(image_file.read(4), byteorder='big')
    column = int.from_bytes(image_file.read(4), byteorder='big')

    train_label = [0 for _ in range(number)]
    train_image = [[[0 for _ in range(column)] for _ in range(row)] for _ in range(number)]

    for i in range(number):
        train_label[i] = label_file.read(1)[0]
        for j in range(row):
            for k in range(column):
                train_image[i][j][k] = int(image_file.read(1)[0] / 128)

    label_file.close()
    image_file.close()

    # 將 train_image 展平以符合 EM 函數的需求
    train_image = np.array(train_image).reshape(number, -1)  # 將每張圖片展平成一維數組

    return train_label, train_image, number

def EM(n, train_image):

    c = np.array([0.1] * 10)  
    p = np.random.rand(10, 28*28)

    result = ""
    dC = 100
    dP = 100
    count = 0

    while dC > 0.01 and dP > 0.01:
        count += 1
        print(f'迭代次數 = {count} \n')
        # E-step
        w = np.zeros((n, 10))
        for i in range(n):
            for j in range(10):
                log_c = np.log(c[j])  
                log_p = np.sum(train_image[i] * np.log(p[j] + 1e-8)) + np.sum((1 - train_image[i]) * np.log(1 - p[j] + 1e-8))
                w[i][j] = log_c + log_p
            w[i] = np.exp(w[i] - max(w[i]))
            w[i] /= np.sum(w[i])

        # M-step
        c_new = np.zeros(10)
        for j in range(10):
            c_new[j] = np.sum(w[:, j]) / n  # 更新每個類別的機率
        p_new = np.dot(np.transpose(w), train_image)  # 使用 np.transpose 提高性能
        for j in range(10):
            p_new[j] /= np.sum(w[:, j]) + 1e-8  # 避免除以零的問題

        dc = np.linalg.norm(c - c_new)
        dp = np.linalg.norm(p - p_new)
        c = c_new
        p = p_new

        for i in range(10):
            result += f"class {i}:\n"
            for row in range(28):
                for column in range(28):
                    result += "1 " if (p[i][row * 28 + column] > 0.5) else "0 "
                result += "\n"
            result += "\n"
        result += f"No. of Iteration: {count}, Difference: {dc + dp:<15.10f}\n\n"
        result += "------------------------------------------------------------\n\n"
        if count > 20:
            break
    print(f'結束迭代 \n')    
    return w, c, p, result, count



def Test(n, c, p, train_image, label, count, initial_w):
    w = initial_w  
    predict = np.zeros(n, dtype=int)
    
    for i in range(n):
        predict[i] = np.argmax(w[i])


    result = ""
    confusion_matrix = np.zeros((10, 2, 2), dtype=int)

    for i in range(n):
        if predict[i] == label[i]:
            confusion_matrix[label[i]][0][0] += 1  # TP
        else:
            confusion_matrix[label[i]][0][1] += 1  # FN
            confusion_matrix[predict[i]][1][0] += 1  # FP

        for j in range(10):
            if j != label[i] and j != predict[i]:
                confusion_matrix[j][1][1] += 1  # TN

    for i in range(10):
        result += f"labeled class {i}:\n"
        for j in range(28):
            for k in range(28):
                result += "1 " if (p[i][j * 28 + k] > 0.5) else "0 "
            result += "\n"
        result += "\n"
    
    TP = np.zeros(10, dtype=int)
    TN = np.zeros(10, dtype=int)
    FP = np.zeros(10, dtype=int)
    FN = np.zeros(10, dtype=int)

    for i in range(10):
        TP[i] = confusion_matrix[i][0][0]
        FP[i] = confusion_matrix[i][0][1]
        FN[i] = confusion_matrix[i][1][0]
        TN[i] = confusion_matrix[i][1][1]

        result += f"Confusion Matrix {i}:\n"
        result += f"                Predict number {i}  Predict not number {i}\n"
        result += f"Is number {i}     {TP[i]}                 {FN[i]}\n"
        result += f"Isn't number {i}  {FP[i]}                 {TN[i]}\n\n"

        sensitivity = TP[i] / (TP[i] + FN[i]) if (TP[i] + FN[i]) > 0 else 0
        specificity = TN[i] / (TN[i] + FP[i]) if (TN[i] + FP[i]) > 0 else 0
        result += f"Sensitivity (Successfully predict number {i}): {sensitivity:.4f}\n"
        result += f"Specificity (Successfully predict not number {i}): {specificity:.4f}\n"
        result += "------------------------------------------------------------\n\n"

    total_correct = np.sum(TP)   
    error_rate = 1 - (total_correct / n) 
    result += f"Total iteration to converge: {count}\n"
    result += f"Total error rate: {error_rate:.4f}\n"     
        
    return result

if __name__ == "__main__":
    # load
    print("Loading...")
    train_label, train_image, n = load()

    # EM algorithm
    print("EM")
    w, c, p, result, count = EM(n, train_image)

    # Test
    print("Testing...")
    result += Test(n, c, p, train_image, train_label, count, w)

    with open(r"C:\Users\yoush\Desktop\機器學習\HW4\result.txt", 'w') as resultFile:
        resultFile.write(result)
        os.startfile('result.txt')
