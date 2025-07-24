# %% part a.b
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os


def Gram_matrix(data_s, data_c, gamma_s=0.001, gamma_c=0.001):
    kernel_s = np.exp(-gamma_s * cdist(data_s, data_s, 'sqeuclidean'))
    kernel_c = np.exp(-gamma_c * cdist(data_c, data_c, 'sqeuclidean'))
    kernel = kernel_s * kernel_c
    return kernel


def initial(kernel, k):
    n = kernel.shape[0]
    center = np.random.choice(n, size=k, replace=False)
    mean = kernel[center, :]
    return mean


def k_means(kernel, k, max_iterations=100):
    mean = initial(kernel, k)
    diff = 10
    history = []  
    iteration = 0

    while diff > 1e-6 and iteration < max_iterations:
        # E-step
        c = np.zeros(kernel.shape[0], dtype=int)
        for i in range(kernel.shape[0]):
         
            distances = [np.linalg.norm(kernel[i] - mean[j]) for j in range(k)]
            c[i] = np.argmin(distances)
        
        history.append(c.copy()) 
        
        # M-step
        previous_mean = mean.copy()
        mean = np.zeros((k, kernel.shape[1]), dtype=kernel.dtype)
        counters = np.zeros(k)
        
        for i in range(kernel.shape[0]):
            mean[c[i]] += kernel[i]
            counters[c[i]] += 1
        
        for i in range(k):
            if counters[i] > 0:
                mean[i] /= counters[i]
            else:
                
                mean[i] = kernel[np.random.randint(kernel.shape[0])]
        
        diff = np.linalg.norm(mean - previous_mean)  
        iteration += 1

    return history

def Laplacian(kernel, cut):
    W = kernel
    D = np.diag(np.sum(W, axis=1))
    L = D - W 
    if cut == 0:
     
        D_sqrt_inv = np.diag(1/np.diag(np.sqrt(D)))
        L = D_sqrt_inv @ L @ D_sqrt_inv
    return L


def eigen(L, k, cut):
    eigval, eigvec = np.linalg.eig(L)

    sorted_index = np.argsort(eigval)

 
    U = eigvec[:, sorted_index[1: k+1]] 
    if cut ==  0: 
      
        U /= np.sqrt(np.sum(np.power(U, 2), axis=1)).reshape(-1,1)
    return U


# %%
import os
if __name__ == "__main__":
    part = input("Enter the part (a or b): ")

    mode = int(input("Enter the mode: 0 for kernel k-means, 1 for spectral clustering: "))

        
    k = int(input("Enter the number of clusters (e.g., 2, 3, 4): "))
    image_number = int(input("Enter the number of images (1 or 2): "))
    
    if image_number == 1:
        image = Image.open(r"C:\Users\yoush\Desktop\機器學習\HW6\image1.png", 'r')
        data = np.array(image)  
        data_c = data.reshape((-1, 3))
        data_s = np.array([(i, j) for i in range(data.shape[0]) for j in range(data.shape[1])])  # 空間座標
    else:
        image = Image.open(r"C:\Users\yoush\Desktop\機器學習\HW6\image2.png", 'r')
        data = np.array(image)  
        data_c = data.reshape((-1, 3))
  
        data_s = np.array([(i, j) for i in range(data.shape[0]) for j in range(data.shape[1])])  # 空間座標
    
    def animate(i):
        result = history[i].reshape((data.shape[0], data.shape[1]))
        img.set_array(result)
        plt.title(f"Iteration {i+1}")
        return img
    
    
    if mode == 0:
        kernel = Gram_matrix(data_s, data_c)
        history = k_means(kernel, k)
        fig, ax = plt.subplots()
        img = ax.imshow(history[0].reshape((data.shape[0], data.shape[1])), cmap='viridis')
        ani = FuncAnimation(fig, animate, frames=len(history), interval=200)
        save_path = f"C:/Users/yoush/Desktop/機器學習/HW6/gif/part_{part}/image_{image_number}/k_means_k ={k}.gif"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ani.save(save_path, writer="pillow")
        plt.figure()
        plt.imshow(history[-1].reshape((data.shape[0], data.shape[1])), cmap='viridis')
        plt.title(f'k_means for k = {k}')
        save_path = fr"C:\Users\yoush\Desktop\機器學習\HW6\png\part_{part}\image_{image_number}\k_means_k ={k}_final_image.png"  # 这里修改为你希望保存的路径
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)  
        plt.close()  
    if mode == 1:
        cut = int(input("Enter the cut: 0 for normalize cut and 1 for ratio cut: ")) 
        kernel = Gram_matrix(data_s, data_c)
        L = Laplacian(kernel, cut)
        U = eigen(L, k, cut)
        history = k_means(U, k)
        fig, ax = plt.subplots()
        img = ax.imshow(history[0].reshape((data.shape[0], data.shape[1])), cmap='viridis')
        ani = FuncAnimation(fig, animate, frames=len(history), interval=200)
        if cut == 0:
            cc = "normalized"
        else:
            cc = "ratio"
            
        save_path = f"C:/Users/yoush/Desktop/機器學習/HW6/gif/part_{part}/image_{image_number}/spectral_clustering({cc})_k ={k}.gif"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ani.save(save_path, writer="pillow")
        plt.figure()
        plt.imshow(history[-1].reshape((data.shape[0], data.shape[1])), cmap='viridis')
        plt.title(f'spectral clustering({cc}) for k = {k}')
        save_path = fr"C:\Users\yoush\Desktop\機器學習\HW6\png\part_{part}\image_{image_number}\spectral_clustering({cc})_k ={k}_final_image.png"  
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)  
        plt.close()   

#%% part c
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os



def Gram_matrix(data_s, data_c, gamma_s=0.001, gamma_c=0.001):
    kernel_s = np.exp(-gamma_s * cdist(data_s, data_s, 'sqeuclidean'))
    kernel_c = np.exp(-gamma_c * cdist(data_c, data_c, 'sqeuclidean'))
    kernel = kernel_s * kernel_c
    return kernel


def initial(kernel, k):
    n = kernel.shape[0]
    centers = []
    
    
    first_center = np.random.randint(n)
    centers.append(first_center)
    
  
    for _ in range(1, k):
     
        distances = np.min([np.linalg.norm(kernel - kernel[center], axis=1) ** 2 for center in centers], axis=0)
      
        probabilities = distances / np.sum(distances)
        next_center = np.random.choice(n, p=probabilities)
        centers.append(next_center)
    
    mean = kernel[centers, :]
    return mean

def k_means(kernel, k, max_iterations=100):
    mean = initial(kernel, k)
    diff = 10
    history = []  # 儲存每次分群結果
    iteration = 0

    while diff > 1e-6 and iteration < max_iterations:
        # E-step
        c = np.zeros(kernel.shape[0], dtype=int)
        for i in range(kernel.shape[0]):
        
            distances = [np.linalg.norm(kernel[i] - mean[j]) for j in range(k)]
            c[i] = np.argmin(distances)
        
        history.append(c.copy())  
        
        # M-step
        previous_mean = mean.copy()
        mean = np.zeros((k, kernel.shape[1]), dtype=kernel.dtype)
        counters = np.zeros(k)
        
        for i in range(kernel.shape[0]):
            mean[c[i]] += kernel[i]
            counters[c[i]] += 1
        
        for i in range(k):
            if counters[i] > 0:
                mean[i] /= counters[i]
            else:
             
                mean[i] = kernel[np.random.randint(kernel.shape[0])]

        diff = np.linalg.norm(mean - previous_mean) 
        iteration += 1

    return history

def Laplacian(kernel, cut):
    W = kernel
    D = np.diag(np.sum(W, axis=1))
    L = D - W  
    if cut == 0:
     
        D_sqrt_inv = np.diag(1/np.diag(np.sqrt(D)))
        L = D_sqrt_inv @ L @ D_sqrt_inv
    return L

def eigen(L, k, cut):
    eigval, eigvec = np.linalg.eig(L)

    sorted_index = np.argsort(eigval)

   
    U = eigvec[:, sorted_index[1: k+1]] 
    if cut == 0: 

        U /= np.sqrt(np.sum(np.power(U, 2), axis=1)).reshape(-1, 1)
    return U




# %%

if __name__ == "__main__":
    part = "c"

    mode = int(input("Enter the mode: 0 for kernel k-means, 1 for spectral clustering: "))

        
    k = int(input("Enter the number of clusters (e.g., 2, 3, 4): "))
    image_number = int(input("Enter the number of images (1 or 2): "))

    if image_number == 1:
        image = Image.open(r"C:\Users\yoush\Desktop\機器學習\HW6\image1.png", 'r')
        data = np.array(image)
        data_c = data.reshape((-1, 3))  # 3 color channels
        data_s = np.array([(i, j, data[i, j, 0], data[i, j, 1], data[i, j, 2]) for i in range(data.shape[0]) for j in range(data.shape[1])])
    else:
        image = Image.open(r"C:\Users\yoush\Desktop\機器學習\HW6\image2.png", 'r')
        data = np.array(image)
        data_c = data.reshape((-1, 3))  # 3 color channels
        data_s = np.array([(i, j, data[i, j, 0], data[i, j, 1], data[i, j, 2]) for i in range(data.shape[0]) for j in range(data.shape[1])])

    def animate(i):
        result = history[i].reshape((data.shape[0], data.shape[1]))
        img.set_array(result)
        plt.title(f"Iteration {i + 1}")
        return img
    

    if mode == 1:
        cut = 1
        kernel = Gram_matrix(data_s, data_c)
        L = Laplacian(kernel, cut)
        U = eigen(L, k, cut)
        history = k_means(U, k)
         
        
        fig, ax = plt.subplots()
        img = ax.imshow(history[0].reshape((data.shape[0], data.shape[1])), cmap='viridis')
        ani = FuncAnimation(fig, animate, frames=len(history), interval=200)
        if cut == 0:
            cc = "normalized"
        else:
            cc = "ratio"
            
        save_path = f"C:/Users/yoush/Desktop/機器學習/HW6/gif/part_c/image_{image_number}/spectral_clustering({cc})_k ={k}.gif"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ani.save(save_path, writer="pillow")
        plt.figure()
        plt.imshow(history[-1].reshape((data.shape[0], data.shape[1])), cmap='viridis')
        plt.title(f'spectral clustering({cc}) for k = {k}')
        save_path = fr"C:\Users\yoush\Desktop\機器學習\HW6\png\part_{part}\image_{image_number}\spectral_clustering({cc})_k ={k}_final_image.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)  
    
    def visualize_eigenspace(U, c, k, cc, image_number):
        U = np.real(U)
        c = np.real(c)
        
        if U.shape[1] > 2:  
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        

            ax.scatter(U[:, 0], U[:, 1], U[:, 2], c=c, cmap='viridis')
        

            ax.set_title(f'Coordinates in the Eigenspace_spectral clustering({cc})(k={k})')
            ax.set_xlabel('dim 1')
            ax.set_ylabel('dim 2')
            ax.set_zlabel('dim 3')
        
            save_path = fr"C:\Users\yoush\Desktop\機器學習\HW6\png\part_d\image_{image_number}\spectral_clustering({cc})_k ={k}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path) 
            plt.show()
        else:  
            plt.figure()
            colors = ['r', 'g', 'b', 'y', 'c', 'm', 'b']  
            for i in range(k):
                cluster_points = U[c == i]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i % len(colors)], label=f'Cluster {i}')
            
            plt.title(f'Coordinates in the Eigenspace_spectral clustering({cc})(k={k})')
            plt.xlabel('dim 1')
            plt.ylabel('dim 2')
            plt.legend()
            
            save_path = fr"C:\Users\yoush\Desktop\機器學習\HW6\png\part_d\image_{image_number}\spectral_clustering({cc})_k ={k}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.show()


        if k == 2 or k == 3:
            c = history[-1]  
            visualize_eigenspace(U, c, k, cc, image_number)
            
            
# %%

import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import os



def Gram_matrix(data_s, data_c, gamma_s=0.001, gamma_c=0.001):
    kernel_s = np.exp(-gamma_s * cdist(data_s, data_s, 'sqeuclidean'))
    kernel_c = np.exp(-gamma_c * cdist(data_c, data_c, 'sqeuclidean'))
    kernel = kernel_s * kernel_c
    return kernel



def stratified_initialization(kernel, k):
    n = kernel.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    partitions = np.array_split(indices, k)
    return kernel[np.array([np.random.choice(partition) for partition in partitions])] 



def density_initialization(kernel, k):
    densities = np.sum(kernel, axis=1) 
    indices = np.argsort(densities)[-k:]  
    return kernel[indices]



def max_min_distance_initialization(kernel, k):
    n = kernel.shape[0]
    centers = [np.random.randint(n)]  
    for _ in range(k - 1):
        distances = np.min(cdist(kernel[centers], kernel), axis=0)
        next_center = np.argmax(distances)
        centers.append(next_center)
    return kernel[centers]




def initial(kernel, k, init_method):

    if init_method == "stratified":
        return stratified_initialization(kernel, k)
    elif init_method == "density":
        return density_initialization(kernel, k)
    elif init_method == "max_min":
        return max_min_distance_initialization(kernel, k)



def k_means(kernel, k, max_iterations=100, init_method="random"):
    mean = initial(kernel, k, init_method)
    diff = 10
    history = []  
    iteration = 0

    while diff > 1e-6 and iteration < max_iterations:
        c = np.zeros(kernel.shape[0], dtype=int)
        for i in range(kernel.shape[0]):
            
            distances = [np.linalg.norm(kernel[i] - mean[j]) for j in range(k)]
            c[i] = np.argmin(distances)
        
        history.append(c.copy())  
        
        previous_mean = mean.copy()
        mean = np.zeros((k, kernel.shape[1]), dtype=kernel.dtype)
        counters = np.zeros(k)
        
        for i in range(kernel.shape[0]):
            mean[c[i]] += kernel[i]
            counters[c[i]] += 1
        
        for i in range(k):
            if counters[i] > 0:
                mean[i] /= counters[i]
            else:
    
                mean[i] = kernel[np.random.randint(kernel.shape[0])]
        
        diff = np.linalg.norm(mean - previous_mean)  
        iteration += 1

    return history


def Laplacian(kernel, cut):
    W = kernel
    D = np.diag(np.sum(W, axis=1))
    L = D - W 
    if cut == 0:
        
        D_sqrt_inv = np.diag(1 / np.diag(np.sqrt(D)))
        L = D_sqrt_inv @ L @ D_sqrt_inv
    return L



def eigen(L, k, cut):
    eigval, eigvec = np.linalg.eig(L)

    sorted_index = np.argsort(eigval)

    U = eigvec[:, sorted_index[1: k + 1]]  
    if cut == 0: 
      
        U /= np.sqrt(np.sum(np.power(U, 2), axis=1)).reshape(-1, 1)
    return U



def animate(i, history, data, ax):
    result = history[i].reshape((data.shape[0], data.shape[1]))
    ax.clear()
    ax.imshow(result, cmap='viridis')
    ax.set_title(f"Iteration {i+1}")
    return ax,


if __name__ == "__main__":
    part = "e"

    mode = int(input("Enter the mode: 0 for kernel k-means, 1 for spectral clustering: "))

    k = int(input("Enter the number of clusters (e.g., 2, 3, 4): "))
    image_number = int(input("Enter the number of images (1 or 2): "))
    
    if image_number == 1:
        image = Image.open(r"C:\Users\yoush\Desktop\機器學習\HW6\image1.png", 'r')
        data = np.array(image)  
        data_c = data.reshape((-1, 3))
        data_s = np.array([(i, j) for i in range(data.shape[0]) for j in range(data.shape[1])])  # 空間座標
    else:
        image = Image.open(r"C:\Users\yoush\Desktop\機器學習\HW6\image2.png", 'r')
        data = np.array(image)  
        data_c = data.reshape((-1, 3))
        data_s = np.array([(i, j) for i in range(data.shape[0]) for j in range(data.shape[1])])  # 空間座標
    
    if mode == 0:
        kernel = Gram_matrix(data_s, data_c)
        history = k_means(kernel, k)
        fig, ax = plt.subplots()
        ani = FuncAnimation(fig, animate, frames=len(history), fargs=(history, data, ax), interval=200)
        save_path = r"C:\Users\yoush\Desktop\機器學習\HW6\gif\part_e\image_{image_number}\spectral\k_means_k ={k}.gif"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ani.save(save_path, writer="pillow")
        plt.figure()
        plt.imshow(history[-1].reshape((data.shape[0], data.shape[1])), cmap='viridis')
        plt.title(f'k_means for k = {k}')
        save_path = fr"C:\Users\yoush\Desktop\機器學習\HW6\png\part_{part}\image_{image_number}\spectral\k_means_k ={k}_final_image.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()  
    
    
    if mode == 1:
        cut = int(input("Enter the cut: 0 for normalize cut and 1 for ratio cut: ")) 
        kernel = Gram_matrix(data_s, data_c)
        L = Laplacian(kernel, cut)
        U = eigen(L, k, cut)
        history = k_means(U, k)
        fig, ax = plt.subplots()
        ani = FuncAnimation(fig, animate, frames=len(history), fargs=(history, data, ax), interval=200)
        if cut == 0:
            cc = "normalized"
        else:
            cc = "ratio"
            
        save_path = f"C:/Users/yoush/Desktop/機器學習/HW6/gif/part_{part}/image_{image_number}/spectral_clustering({cc})_k ={k}.gif"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ani.save(save_path, writer="pillow")
        plt.figure()
        plt.imshow(history[-1].reshape((data.shape[0], data.shape[1])), cmap='viridis')
        plt.title(f'spectral clustering({cc}) for k = {k}')
        save_path = fr"C:\Users\yoush\Desktop\機器學習\HW6\png\part_{part}\image_{image_number}\spectral_clustering({cc})_k ={k}_final_image.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

