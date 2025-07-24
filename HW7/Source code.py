import os
from PIL import Image
from scipy.spatial.distance import cdist
from matplotlib import pyplot
import numpy as np
#%%  1
# train_load
train_path = "C:\\Users\\yoush\\Desktop\\機器學習\\HW7\\Yale_Face_Database\\Yale_Face_Database\\Training\\"
files = os.listdir(train_path)

train_image = []
train_subclass = 9
train_label = [[i] * train_subclass for i in range(15)]

for file in files:

    file_path = os.path.join(train_path, file)
    image = Image.open(file_path)
    train_image.append(np.array(image).reshape(-1))  

train_image = np.array(train_image)
train_label = np.array(train_label).reshape(-1)


# test_load
test_path = "C:\\Users\\yoush\\Desktop\\機器學習\\HW7\\Yale_Face_Database\\Yale_Face_Database\\Testing\\"
files = os.listdir(test_path)

test_image = []
test_subclass = 2
test_label = [[i] * test_subclass for i in range(15)]

for file in files:

    file_path = os.path.join(test_path, file)
    image = Image.open(file_path)
    test_image.append(np.array(image).reshape(-1))  

test_image = np.array(test_image)
test_label = np.array(test_label).reshape(-1)

#%%  1-Part 1

def PCA(train_image, train_label):
    mean = np.mean(train_image, axis=0)
    center = train_image - mean

    S = np.cov(train_image, bias=True)
    eigval, eigvec = np.linalg.eig(S)
    index = np.argsort(eigval)[::-1]
    eigval= eigval[index].real
    eigvec = eigvec[:,index].real
    
    # first 25 eigenfaces 
    transform = center.T @ eigvec
    fig, axes = pyplot.subplots(5, 5, figsize=(10, 10))
    
    for i, ax in enumerate(axes.flat):
        ax.axis("off") 
        ax.imshow(transform[:, i].reshape(231, 195), cmap="gray")
    pyplot.show()
    
    # reconstruct    
    z = transform.T @ center.T
    reconstruct = transform @ z + mean.reshape(-1, 1)
    index = np.random.choice(135, 10, replace=False) # 9*15=135
    
    fig, axes = pyplot.subplots(2, 10, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        ax.axis("off") 
        if i < 10:
            ax.imshow(train_image[index][i].reshape(231, 195), cmap="gray")
        else:
            ax.imshow(reconstruct[:, index][:, i - 10].reshape(231, 195), cmap="gray")
    pyplot.show()

    return transform, z

def LDA(pca_transform, pca_z, train_image, train_label):
    
    mean = np.mean(pca_z, axis=1)
    N = pca_z.shape[0] 

    S_within = np.zeros((N, N))
    for i in range(15):
        S_within += np.cov(pca_z[:, i*9:i*9+9], bias=True)

    S_between = np.zeros((N, N))
    for i in range(15):
        class_mean = np.mean(pca_z[:, i*9:i*9+9], axis=1).T
        S_between += 9 * (class_mean - mean) @ (class_mean - mean).T

    S = np.linalg.inv(S_within) @ S_between
    eigval, eigvec = np.linalg.eig(S)
    index = np.argsort(eigval)[::-1]
    eigval= eigval[index].real
    eigvec = eigvec[:,index].real

    
    # first 25 fisherfaces
    transform = pca_transform @ eigvec
    fig, axes = pyplot.subplots(5, 5, figsize=(10, 10))
    
    for i, ax in enumerate(axes.flat):
        ax.axis("off") 
        ax.imshow(transform[:, i].reshape(231, 195), cmap="gray")
    pyplot.show()
    mean = np.mean(train_image, axis=0)
    center = train_image - mean
    z = transform.T @ center.T
    
    # reconstruct 
    reconstruct = transform @ z + mean.reshape(-1, 1)
    fig, axes = pyplot.subplots(2, 10, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        ax.axis("off") 
        if i < 10:
            ax.imshow(train_image[index][i].reshape(231, 195), cmap="gray")
        else:
            ax.imshow(reconstruct[:, index][:, i - 10].reshape(231, 195), cmap="gray")
    pyplot.show()
    
    fisher_z = eigvec.T @ pca_z
    return fisher_z, transform

pca_transform, pca_z = PCA(train_image, train_label)
fisher_z, fisher_transform = LDA(pca_transform, pca_z, train_image, train_label)

# %% 1-part2

def face_recognition(fisher_z, fisher_transform, train_image, train_label, test_image, test_label, k=3):
 
    train_proj = fisher_z.T
    
   
    mean = np.mean(train_image, axis=0)
    test_centered = test_image - mean
    test_proj = fisher_transform.T @ test_centered.T
    
    # k-NN 
    distances = cdist(test_proj.T, train_proj, metric="euclidean")
    neighbors = np.argsort(distances, axis=1)[:, :k]
    pred_labels = []

    for i in range(test_proj.shape[1]):
        neighbor_labels = train_label[neighbors[i]]
        pred_labels.append(np.argmax(np.bincount(neighbor_labels)))  
    
    correct_count = sum([1 for i in range(len(test_label)) if test_label[i] == pred_labels[i]])
    accuracy = correct_count / len(test_label)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return pred_labels

predicted_labels = face_recognition(fisher_z, fisher_transform, train_image, train_label, test_image, test_label, k=3)

# %% 1-part3

def kernel_PCA(train_image, train_label, kernel, k=3): 
    
    mean = np.mean(train_image, axis=0)
    center = train_image - mean
    
    
    N = train_image.shape[0]
    if kernel == "linear":
        K = train_image @ train_image.T 
    elif kernel == "poly":
        K = (train_image @ train_image.T) ** 3
    elif kernel == "RBF":
        K = np.exp(-0.01 * cdist(train_image, train_image, 'sqeuclidean'))  
     
    one_N = np.ones((N, N)) / N
    K_centered = K - one_N @ K - K @ one_N + one_N @ K @ one_N
    
    eigval, eigvec = np.linalg.eig(K_centered)
    index = np.argsort(eigval)[::-1]
    eigval= eigval[index].real
    eigvec = eigvec[:,index].real

    transform = center.T @ eigvec

    z = transform.T @ center.T
    
    return transform, z

def kernel_LDA(pca_transform, pca_z, train_image, train_label):
    
    mean = np.mean(pca_z, axis=1)
    N = pca_z.shape[0] 

    
    S_within = np.zeros((N, N))
    for i in range(15):  
        S_within += np.cov(pca_z[:, i*9:i*9+9], bias=True)

 
    S_between = np.zeros((N, N))
    for i in range(15):
        class_mean = np.mean(pca_z[:, i*9:i*9+9], axis=1).T
        S_between += 9 * (class_mean - mean) @ (class_mean - mean).T

   
    S = np.linalg.inv(S_within) @ S_between
    eigval, eigvec = np.linalg.eig(S)
    index = np.argsort(eigval)[::-1]
    eigval = eigval[index].real
    eigvec = eigvec[:, index].real
    
    
    fisher_transform = pca_transform @ eigvec
   
    fisher_z = eigvec.T @ pca_z
    
    return fisher_z, fisher_transform


def face_recognition(fisher_z, fisher_transform, train_image, train_label, test_image, test_label, kernel, k=3):
   
    train_proj = fisher_z.T
    mean = np.mean(train_image, axis=0)
    test_centered = test_image - mean
    test_proj = fisher_transform.T @ test_centered.T
    
  
    distances = cdist(test_proj.T, train_proj, metric="euclidean")
    neighbors = np.argsort(distances, axis=1)[:, :k]
    pred_labels = []

    for i in range(test_proj.shape[1]):
        neighbor_labels = train_label[neighbors[i]]
        pred_labels.append(np.argmax(np.bincount(neighbor_labels)))  
    
    correct_count = sum([1 for i in range(len(test_label)) if test_label[i] == pred_labels[i]])
    accuracy = correct_count / len(test_label)
    print(f"Kernel: {kernel} | Accuracy: {accuracy * 100:.2f}%")

    return pred_labels




kernel = "linear"
pca_transform, pca_z = kernel_PCA(train_image, train_label, kernel)  
fisher_z, fisher_transform = kernel_LDA(pca_transform, pca_z, train_image, train_label)
predicted_labels = face_recognition(fisher_z, fisher_transform, train_image, train_label, test_image, test_label, kernel, k=3)


kernel = "poly"
pca_transform, pca_z = kernel_PCA(train_image, train_label, kernel)  
fisher_z, fisher_transform = kernel_LDA(pca_transform, pca_z, train_image, train_label)
predicted_labels = face_recognition(fisher_z, fisher_transform, train_image, train_label, test_image, test_label, kernel, k=3)



kernel = "RBF"
pca_transform, pca_z = kernel_PCA(train_image, train_label, kernel)  
fisher_z, fisher_transform = kernel_LDA(pca_transform, pca_z, train_image, train_label)
predicted_labels = face_recognition(fisher_z, fisher_transform, train_image, train_label, test_image, test_label, kernel, k=3)




# %% 2

#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import os
import numpy as np
import pylab
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.colors as mcolors

def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y

def symmetric_sne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0):


    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    Y_history = [(0, Y.copy())]
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))
        Y_history.append((iter + 1, Y.copy()))
        # Compute current value of cost function
        if (iter + 1) % 10 == 0:

            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.
    
    pylab.title('Symmetric-SNE high-dim')
    pylab.hist(P.flatten(),bins=40,log=True, color='skyblue')
    pylab.show()
    
    pylab.title('Symmetric-SNE low-dim')
    pylab.hist(Q.flatten(),bins=40,log=True, color='red')
    pylab.show()
    # Return solution
    
    pylab.title('Symmetric-SNE')
    pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
    pylab.show()
    return Y, Y_history
def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    Y_history = [(0, Y.copy())]
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))
        Y_history.append((iter + 1, Y.copy()))
        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.
    
    pylab.title('t-SNE high-dim')
    pylab.hist(P.flatten(),bins=40,log=True, color='skyblue')
    pylab.show()
    
    pylab.title('t-SNE low-dim')
    pylab.hist(Q.flatten(),bins=40,log=True, color='red')
    pylab.show()
    
    pylab.title('t-SNE')
    pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
    pylab.show()


    return Y, Y_history


def create_animation(Y_history, labels, title, save_dir, filename, axis_range=(-50, 50)):
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter([], [], c=[], cmap='tab10', s=20)

    # Set axis properties
    ax.set_title(title)
    ax.set_xlim(axis_range)
    ax.set_ylim(axis_range)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # Initialization function
    def init():
        scatter.set_offsets(np.c_[[], []])  # Clear points initially
        scatter.set_array([])  # Reset the color array
        return [scatter]  # Return as a list to make it iterable

    # Update function for each frame
    def update(frame):
        iteration, Y = Y_history[frame]  # Unpack the frame data
        scatter.set_offsets(np.c_[Y[:, 0], Y[:, 1]])  # Set new positions
        scatter.set_array(labels)  # Update the color array based on labels
        return [scatter]  # Return as a list to make it iterable

    # Create the animation object, only for the first 100 frames
    frames_to_use = min(100, len(Y_history))  # Ensure it doesn't exceed the length of Y_history
    anim = FuncAnimation(fig, update, init_func=init,
                         frames=frames_to_use,
                         interval=200,  # Frame interval in ms
                         blit=True)

    # Save the animation to the specified file
    writer = PillowWriter(fps=30)
    anim.save(save_path, writer=writer)

    plt.close()  # Close the plot window after saving

    print(f"Animation saved to: {save_path}")
    print(f"Total frames: {frames_to_use}")



# %% ssne 10
print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform symmetric-SNE on your dataset.")
print("Running example on 2,500 MNIST digits...")
X = np.loadtxt(r"C:\Users\yoush\Desktop\機器學習\HW7\tsne_python\tsne_python\mnist2500_X.txt")
labels = np.loadtxt(r"C:\Users\yoush\Desktop\機器學習\HW7\tsne_python\tsne_python\mnist2500_labels.txt")
Y, Y_history_ssne = symmetric_sne(X, 2, 50, 10.0)
save_dir = "C:\\Users\\yoush\\Desktop\\機器學習\\HW7\\ssne\\"
create_animation(Y_history_ssne, labels, "Symmetric SNE(perplexity 10)", save_dir, "ssne_animation(perplexity_10).gif", axis_range=(-80, 80))
# %% ssne 20
print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform symmetric-SNE on your dataset.")
print("Running example on 2,500 MNIST digits...")
X = np.loadtxt(r"C:\Users\yoush\Desktop\機器學習\HW7\tsne_python\tsne_python\mnist2500_X.txt")
labels = np.loadtxt(r"C:\Users\yoush\Desktop\機器學習\HW7\tsne_python\tsne_python\mnist2500_labels.txt")
Y, Y_history_ssne = symmetric_sne(X, 2, 50, 20.0)
save_dir = "C:\\Users\\yoush\\Desktop\\機器學習\\HW7\\ssne\\20\\"
create_animation(Y_history_ssne, labels, "Symmetric SNE(perplexity 20)", save_dir, "ssne_animation(perplexity_20).gif", axis_range=(-50, 50))
# %% ssne 30

print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform symmetric-SNE on your dataset.")
print("Running example on 2,500 MNIST digits...")
X = np.loadtxt(r"C:\Users\yoush\Desktop\機器學習\HW7\tsne_python\tsne_python\mnist2500_X.txt")
labels = np.loadtxt(r"C:\Users\yoush\Desktop\機器學習\HW7\tsne_python\tsne_python\mnist2500_labels.txt")
Y, Y_history_ssne = symmetric_sne(X, 2, 50, 30.0)
save_dir = "C:\\Users\\yoush\\Desktop\\機器學習\\HW7\\ssne\\30\\"
create_animation(Y_history_ssne, labels, "Symmetric SNE(perplexity 30)", save_dir, "ssne_animation(perplexity_30).gif", axis_range=(-30, 30))

# %% ssne 40

print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform symmetric-SNE on your dataset.")
print("Running example on 2,500 MNIST digits...")
X = np.loadtxt(r"C:\Users\yoush\Desktop\機器學習\HW7\tsne_python\tsne_python\mnist2500_X.txt")
labels = np.loadtxt(r"C:\Users\yoush\Desktop\機器學習\HW7\tsne_python\tsne_python\mnist2500_labels.txt")
Y, Y_history_ssne = symmetric_sne(X, 2, 50, 40.0)
save_dir = "C:\\Users\\yoush\\Desktop\\機器學習\\HW7\\ssne\\40\\"
create_animation(Y_history_ssne, labels, "Symmetric SNE(perplexity 40)", save_dir, "ssne_animation(perplexity_40).gif", axis_range=(-20, 20))

# %% ssne 50

print("Run Y = ssne.ssne(X, no_dims, perplexity) to perform symmetric-SNE on your dataset.")
print("Running example on 2,500 MNIST digits...")
X = np.loadtxt(r"C:\Users\yoush\Desktop\機器學習\HW7\tsne_python\tsne_python\mnist2500_X.txt")
labels = np.loadtxt(r"C:\Users\yoush\Desktop\機器學習\HW7\tsne_python\tsne_python\mnist2500_labels.txt")
Y, Y_history_ssne = symmetric_sne(X, 2, 50, 50.0)
save_dir = "C:\\Users\\yoush\\Desktop\\機器學習\\HW7\\ssne\\50\\"
create_animation(Y_history_ssne, labels, "Symmetric SNE(perplexity 50)", save_dir, "ssne_animation(perplexity_50).gif", axis_range=(-20, 20))

# %% tsne 10

print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
print("Running example on 2,500 MNIST digits...")
X = np.loadtxt(r"C:\Users\yoush\Desktop\機器學習\HW7\tsne_python\tsne_python\mnist2500_X.txt")
labels = np.loadtxt(r"C:\Users\yoush\Desktop\機器學習\HW7\tsne_python\tsne_python\mnist2500_labels.txt")
Y, Y_history_ssne = tsne(X, 2, 50, 10.0)
save_dir = "C:\\Users\\yoush\\Desktop\\機器學習\\HW7\\tsne\\10\\"
create_animation(Y_history_ssne, labels, "t-SNE(perplexity 10)", save_dir, "tsne_animation(perplexity_10).gif", axis_range=(-20, 20))
# %% tsne 20
print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform symmetric-SNE on your dataset.")
print("Running example on 2,500 MNIST digits...")
X = np.loadtxt(r"C:\Users\yoush\Desktop\機器學習\HW7\tsne_python\tsne_python\mnist2500_X.txt")
labels = np.loadtxt(r"C:\Users\yoush\Desktop\機器學習\HW7\tsne_python\tsne_python\mnist2500_labels.txt")
Y, Y_history_ssne = tsne(X, 2, 50, 20.0)
save_dir = "C:\\Users\\yoush\\Desktop\\機器學習\\HW7\\tsne\\20\\"
create_animation(Y_history_ssne, labels, "t-SNE(perplexity 20)", save_dir, "tsne_animation(perplexity_20).gif", axis_range=(-20, 20))
# %% tsne 30

print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform symmetric-SNE on your dataset.")
print("Running example on 2,500 MNIST digits...")
X = np.loadtxt(r"C:\Users\yoush\Desktop\機器學習\HW7\tsne_python\tsne_python\mnist2500_X.txt")
labels = np.loadtxt(r"C:\Users\yoush\Desktop\機器學習\HW7\tsne_python\tsne_python\mnist2500_labels.txt")
Y, Y_history_ssne = tsne(X, 2, 50, 30.0)
save_dir = "C:\\Users\\yoush\\Desktop\\機器學習\\HW7\\tsne\\30\\"
create_animation(Y_history_ssne, labels, "t-SNE(perplexity 30)", save_dir, "tsne_animation(perplexity_30).gif", axis_range=(-20, 20))

# %% tsne 40

print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform symmetric-SNE on your dataset.")
print("Running example on 2,500 MNIST digits...")
X = np.loadtxt(r"C:\Users\yoush\Desktop\機器學習\HW7\tsne_python\tsne_python\mnist2500_X.txt")
labels = np.loadtxt(r"C:\Users\yoush\Desktop\機器學習\HW7\tsne_python\tsne_python\mnist2500_labels.txt")
Y, Y_history_ssne = tsne(X, 2, 50, 40.0)
save_dir = "C:\\Users\\yoush\\Desktop\\機器學習\\HW7\\tsne\\40\\"
create_animation(Y_history_ssne, labels, "t-SNE(perplexity 40)", save_dir, "tsne_animation(perplexity_40).gif", axis_range=(-20, 20))

# %% tsne 50

print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform symmetric-SNE on your dataset.")
print("Running example on 2,500 MNIST digits...")
X = np.loadtxt(r"C:\Users\yoush\Desktop\機器學習\HW7\tsne_python\tsne_python\mnist2500_X.txt")
labels = np.loadtxt(r"C:\Users\yoush\Desktop\機器學習\HW7\tsne_python\tsne_python\mnist2500_labels.txt")
Y, Y_history_ssne = tsne(X, 2, 50, 50.0)
save_dir = "C:\\Users\\yoush\\Desktop\\機器學習\\HW7\\tsne\\50\\"
create_animation(Y_history_ssne, labels, "t-SNE(perplexity 50)", save_dir, "tsne_animation(perplexity_50).gif", axis_range=(-20, 20))

















