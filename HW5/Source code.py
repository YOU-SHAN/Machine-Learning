import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from libsvm.svmutil import *
import pandas as pd
from scipy.spatial.distance import cdist

# %% 1.
df = open(r"C:\Users\yoush\Desktop\機器學習\HW5\data\input.data","r")

file_path = r"C:\Users\yoush\Desktop\機器學習\HW5\data\input.data"
x = []
y = []
with open(file_path, "r") as df:
    lines = df.readlines() 
points = []
for line in lines:
    values = list(map(float, line.split()))  
    for i in range(0, len(values), 2):
        x_point, y_point = values[i], values[i+1]
        x.append(x_point)
        y.append(y_point)

def kernel(x1, x2, sigma , alpha , l):
    diff = np.expand_dims(x1, axis=1) - np.expand_dims(x2, axis=0)  
    return sigma**2 * (1 + np.power(diff, 2) / (2 * alpha * l ** 2)) ** (-alpha)


def Gaussian(x, beta, sigma, alpha , l):
    k = kernel(x, np.transpose(x), sigma, alpha, l)
    c = k + (1 / beta) * np.eye(len(x))
    return c

def predict(x, y, c, beta, sigma, alpha, l):
    x_star = np.linspace(-60, 60, 1000)
    pred_mean = np.zeros(1000)
    pred_var = np.zeros(1000)
    c_inv = np.linalg.inv(c)
    for i in range(1000):
        k_x_xs = kernel(x, [x_star[i]], sigma, alpha, l)  
        pred_mean[i] = np.transpose(k_x_xs) @ c_inv @ y  
        
        k_xs_xs = kernel([x_star[i]], [x_star[i]], sigma, alpha, l) + (1/beta) 
        pred_var[i] = np.abs(k_xs_xs - np.transpose(k_x_xs) @ c_inv @ k_x_xs)

    return x_star, pred_mean, pred_var

#task1

sigma = 1
alpha = 1
l = 1
beta = 5
c = Gaussian(x, beta, sigma, alpha , l)
x_star, pred_mean, pred_var = predict(x, y, c, beta, sigma, alpha, l)

plt.plot(x, y, 'bo',markersize=3)
plt.plot(x_star, pred_mean, 'k-')
plt.fill_between(x_star, pred_mean + 2 * (pred_var)**(1/2), pred_mean - 2 * (pred_var)**(1/2), facecolor = 'pink')
plt.xlim(-60, 60)
plt.title(f"normal\nsigma = {sigma:.2f}, alpha = {alpha:.2f}, l = {l:.2f}")
plt.show()

#task2
def marginal_likelihood(theta, x, y, beta):
    theta = theta.ravel()  
    c = Gaussian(x, beta, theta[0], theta[1], theta[2]) 
    L = 0.5 * (np.log(np.linalg.det(c)) + np.transpose(y) @ np.linalg.inv(c) @ y + np.log(2*np.pi) * len(x)) 
    return L.ravel()  

sigma = 1
alpha = 1
l = 1
beta = 5


optimize = minimize(marginal_likelihood, [sigma, alpha, l],
                   bounds = ((1e-6, 1e6),  # the range of sigma
                            (1e-6, 1e2),   # the range of alpha 
                            (1e-6, 1e2)),  # the range of l 
                    args=(x, y, beta))  
sigma_hat = optimize.x[0]
alpha_hat = optimize.x[1]
l_hat = optimize.x[2]


c_hat = Gaussian(x, beta, sigma_hat, alpha_hat, l_hat)
x_star, pred_mean, pred_var = predict(x, y, c_hat, beta, sigma_hat, alpha_hat, l_hat)


plt.plot(x, y, 'bo',markersize=3)
plt.plot(x_star, pred_mean, 'k-')
plt.fill_between(x_star, pred_mean + 2 * (pred_var)**(1/2), pred_mean - 2 * (pred_var)**(1/2), facecolor = 'pink')
plt.xlim(-60, 60)
plt.title(f"optimize\nsigma = {sigma_hat:.2f}, alpha = {alpha_hat:.2f}, l = {l_hat:.2f}")
plt.show()

# %% 2.


train_image_path =  r"C:\Users\yoush\Desktop\機器學習\HW5\data\X_train.csv"
train_label_path = r"C:\Users\yoush\Desktop\機器學習\HW5\data\Y_train.csv"
test_image_path = r"C:\Users\yoush\Desktop\機器學習\HW5\data\X_test.csv"
test_label_path = r"C:\Users\yoush\Desktop\機器學習\HW5\data\Y_test.csv"

train_label = np.array(pd.read_csv(train_label_path, header=None)).ravel().tolist()
test_label = np.array(pd.read_csv(test_label_path, header=None)).ravel().tolist()
train_image = []
df = pd.read_csv(train_image_path, header=None)
for i in range(len(df)):
    train_image.append(df.iloc[i].tolist())

test_image = []
df = pd.read_csv(test_image_path, header=None)
for i in range(len(df)):
    test_image.append(df.iloc[i].tolist())
    # %%
    
# task1

train = svm_problem(train_label, train_image)

# linear
linear_parm = svm_parameter('-q -t 0') 
linear_model = svm_train(train, linear_parm)
# testing
print("linear kernel:")
svm_predict(test_label, test_image, linear_model)
print("\n")

#polynomial
polynomail_parm = svm_parameter('-q -t 1') 
polynomial_model = svm_train(train, polynomail_parm)
# testing
print("polynomial kernel:")
svm_predict(test_label, test_image, polynomial_model)
print("\n")

#RBF
RBF_parm = svm_parameter('-q -t 2')
RBF_model = svm_train(train, RBF_parm)
# testing
print("RBF kernel:")
svm_predict(test_label, test_image, RBF_model)

# %%
# task2

def svm_grid_search(train_label, train_image, test_label, test_image):
    train = svm_problem(train_label, train_image)
    log2c = [i for i in range(-6, 7)]
    log2g = [i for i in range(-6, 7)]
    kernel = int(input("Please select the kernel type by entering the corresponding number: (0: Linear, 1: Polynomial, 2: RBF) "))
    best = 0.0
    best_log2c = 0
    best_log2g = 0

    if kernel == 0:  # Linear kernel
        auc = np.zeros(len(log2c))
        for j in range(len(log2c)):
            parm = f"-q -t {kernel} -v 3 -c {2**log2c[j]}"
            model = svm_train(train, parm)
            auc[j] = model
            if best < model:
                best = model
                best_log2c = log2c[j]

   
        plt.figure(figsize=(8, 6))
        plt.plot(log2c, auc, marker='o')
        plt.xlabel('log2(C)')
        plt.ylabel('Cross Validation Accuracy (%)')
        plt.title("Grid Search (Linear Kernel)")
        plt.grid(True)
        for j in range(len(log2c)):
            plt.text(log2c[j], auc[j], f"{auc[j]:.2f}", ha='center', va='bottom', color='black')
        plt.show()
    else:  # Non-linear kernels (e.g., Polynomial, RBF)
        auc = np.zeros((len(log2g), len(log2c)))
        for i in range(len(log2g)):
            for j in range(len(log2c)):
                parm = f"-q -t {kernel} -v 3 -c {2**log2c[j]} -g {2**log2g[i]}"
                model = svm_train(train, parm)
                auc[i][j] = model
                if best < model:
                    best = model
                    best_log2c = log2c[j]
                    best_log2g = log2g[i]

        plt.figure(figsize=(12, 12))
        plt.imshow(auc, interpolation='nearest', cmap='Blues', origin='lower')
        plt.colorbar(label='Cross Validation Accuracy (%)')
        plt.xticks(range(len(log2c)), [f"{c}" for c in log2c])
        plt.yticks(range(len(log2g)), [f"{g}" for g in log2g])
        plt.xlabel('log2(C)')
        plt.ylabel('log2(Gamma)')
        plt.title(f"Grid Search Heatmap (Kernel={kernel})")
        for i in range(len(log2g)):
            for j in range(len(log2c)):
                plt.text(j, i, f"{auc[i, j]:.2f}", ha='center', va='center', color='pink')
        plt.show()

    parm = f"-q -t {kernel} -c {2**best_log2c}"
    if kernel != 0: 
        parm += f" -g {2**best_log2g}"
    model = svm_train(train, parm)
    _, p_auc, _ = svm_predict(test_label, test_image, model)
    
    return best, best_log2c, best_log2g if kernel != 0 else None, auc, p_auc[0]
# kernel=0
best, best_log2c, best_log2g, auc, p_auc = svm_grid_search(train_label, train_image, test_label, test_image)
# kernel=1
best, best_log2c, best_log2g, auc, p_auc = svm_grid_search(train_label, train_image, test_label, test_image)
# kernel=2
best, best_log2c, best_log2g, auc, p_auc = svm_grid_search(train_label, train_image, test_label, test_image)

# task3

def new_kernel(x1, x2, gamma):
    kernel_linear = x1 @ x2.T
    kernel_RBF = np.exp(-gamma * cdist(x1, x2, 'sqeuclidean'))
    kernel = np.zeros((len(x1), len(x2) + 1))
    kernel[:, 1:] = kernel_linear + kernel_RBF
    kernel[:, :1] = np.arange(len(x1))[:, np.newaxis] + 1
    return kernel


log2c = [i for i in range(-6, 7)]
log2g = [i for i in range(-6, 7)]
best = 0.0
best_log2c = 0
best_log2g = 0

auc = np.zeros((len(log2g), len(log2c)))

for i in range(len(log2g)):
    for j in range(len(log2c)):
        parm = f"-q -t 4 -v 3 -c {2**log2c[j]} -g {2**log2g[i]}"
        kernel = new_kernel(np.array(train_image), np.array(train_image), 2**log2g[i])
        model = svm_train(train_label, [list(row) for row in kernel], parm)
        auc[i][j] = model
        if (best < model):
                best = model
                best_log2g = log2g[i]
                best_log2c = log2c[j]
            
plt.figure(figsize=(12, 12))
plt.imshow(auc, interpolation='nearest', cmap='Blues', origin='lower')
plt.colorbar(label='Cross Validation Accuracy (%)')
plt.xticks(range(len(log2c)), [f"{c}" for c in log2c])
plt.yticks(range(len(log2g)), [f"{g}" for g in log2g])
plt.xlabel('log2(C)')
plt.ylabel('log2(Gamma)')
plt.title(f"Grid Search Heatmap")
for i in range(len(log2g)):
    for j in range(len(log2c)):
        plt.text(j, i, f"{auc[i, j]:.2f}", ha='center', va='center', color='black')
plt.show()
            

parm = f"-q -t 4 -c {2**best_log2c} -g {2**best_log2g}"
best_train_kernel = new_kernel(np.array(train_image), np.array(train_image), 2**best_log2g)
model = svm_train(train_label, [list(row) for row in best_train_kernel], parm)


test_kernel = new_kernel(np.array(test_image), np.array(train_image), 2**best_log2g)
_, p_acc, _ = svm_predict(test_label, [list(row) for row in test_kernel], model)

print(p_acc)