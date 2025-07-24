# %%

import numpy as np
import math 
import os
import matplotlib.pyplot as plt

# %% 1 剖析 IDX 格式的圖片和標籤檔案

def to_int32(byte_data):
    # 將 4 個字節（32 位）轉換為整數，使用大端格式
    return (byte_data[0] << 24) | (byte_data[1] << 16) | (byte_data[2] << 8) | byte_data[3]

def to_uint8(byte_data):
    # 將單個字節（8 位）轉換為無符號整數
    return byte_data[0]

def parse_image_file(image_file_path):
    with open(image_file_path, 'rb') as f:
        print("TRAINING SET IMAGE FILE (train-images.idx3-ubyte)")
        print("offset | type           | value         | description")

        # 读取 magic number (32-bit integer)
        offset = 0
        magic_number = f.read(4)
        magic_number_value = to_int32(magic_number)
        print(f"{offset:04d} | 32-bit integer | 0x{magic_number_value:08X} ({magic_number_value}) | magic number")
        offset += 4

        # 讀取 number of images (32-bit integer)
        num_images = f.read(4)
        num_images_value = to_int32(num_images)
        print(f"{offset:04d} | 32-bit integer | {num_images_value}        | number of images")
        offset += 4

        # 讀取 number of rows (32-bit integer)
        num_rows = f.read(4)
        num_rows_value = to_int32(num_rows)
        print(f"{offset:04d} | 32-bit integer | {num_rows_value}        | number of rows")
        offset += 4

        # 讀取 number of columns (32-bit integer)
        num_cols = f.read(4)
        num_cols_value = to_int32(num_cols)
        print(f"{offset:04d} | 32-bit integer | {num_cols_value}        | number of columns")
        offset += 4
        
        # 用於存儲像素數據的列表
        pixel_data = []
        
        # 逐個像素讀取並存儲
        for image_idx in range(num_images_value):
            for row in range(num_rows_value):
                for col in range(num_cols_value):
                    pixel = f.read(1)  # 讀取 1 個字節
                    if pixel:
                        pixel_value = to_uint8(pixel)
                        pixel_data.append((offset, "unsigned byte", pixel_value, "pixel"))
                        offset += 1

        # 顯示前三個和最後三個像素數據
        if len(pixel_data) > 6:  # 確保有足夠的數據
            for line in pixel_data[:3]:  # 顯示前三個
                print(f"{line[0]:04d} | {line[1]:<15} | {line[2]:03d}          | {line[3]}")
            print("...")  # 中間省略
            for line in pixel_data[-3:]:  # 顯示最後三個
                print(f"{line[0]:04d} | {line[1]:<15} | {line[2]:03d}          | {line[3]}")
        else:
            # 如果數據少於6個，則全部顯示
            for line in pixel_data:
                print(f"{line[0]:04d} | {line[1]:<15} | {line[2]:03d}          | {line[3]}")
                
def parse_label_file(label_file_path):
    with open(label_file_path, 'rb') as f:
        print("\nTRAINING SET LABEL FILE (train-labels.idx1-ubyte)")
        print("offset | type           | value         | description")

        # 讀取 magic number (32-bit integer)
        offset = 0
        magic_number = f.read(4)
        magic_number_value = to_int32(magic_number)
        print(f"{offset:04d} | 32-bit integer | 0x{magic_number_value:08X} ({magic_number_value}) | magic number")
        offset += 4

        # 讀取 number of labels (32-bit integer)
        num_labels = f.read(4)
        num_labels_value = to_int32(num_labels)
        print(f"{offset:04d} | 32-bit integer | {num_labels_value}        | number of labels")
        offset += 4
        labels = []

        for label_idx in range(num_labels_value):
            label = f.read(1)  # 讀取 1 個字節
            label_value = to_uint8(label)
            labels.append((offset, 'unsigned byte', label_value, 'label'))
            offset += 1

        for line in labels[:3]:
            print(f"{line[0]:04d} | {line[1]:<15} | {line[2]:03d}          | {line[3]}")

        if len(labels) > 6:  
            print("...")

        for line in labels[-3:]:
            print(f"{line[0]:04d} | {line[1]:<15} | {line[2]:03d}          | {line[3]}")
                  
            
# %%
image_file_path = 'train-images.idx3-ubyte_'
parse_image_file(image_file_path)

## 上述為資料圖片的結構，其中 value 在 type 為 unsigned byte 中的數字為 0~255 此代表著圖片的灰白值，從第二列可以看出這筆訓練資料有六萬張圖片，也可以由offset的數值進行計算可以驗證此，計算如下:¶
# %%

print("在train image中有:",int((47040015-16+1)/(28*28)),"張圖片") # 除以28*28 理由為 照片大小的長寬

# %% 

label_file_path = 'train-labels.idx1-ubyte_'
parse_label_file(label_file_path)

##上述為資料標籤的結構，其中 value 在 type 為 unsigned byte 中的數字為 0~9 (意即數字零到九) ，從第二列可以看出這筆訓練資料有六萬張圖片，也可以由offset的數值進行計算可以驗證此，計算如下:
    
    
# %%

print("由上述再次驗證，有:",int(60007-8+1),"張圖片")

# %% 載入資料集

def load():
    label_file = open("C:\\Users\\yoush\\Desktop\\機器學習\\HW2\\train-labels.idx1-ubyte_", "rb")
    image_file = open("C:\\Users\\yoush\\Desktop\\機器學習\\HW2\\train-images.idx3-ubyte_", "rb")

    label_file.read(8)
    image_file.read(4)
    number = int.from_bytes(image_file.read(4), byteorder='big')
    row = int.from_bytes(image_file.read(4), byteorder='big')
    column = int.from_bytes(image_file.read(4), byteorder='big')
    
    # 創建一個大小為 number 的列表，初始值為 0
    train_label = [0 for _ in range(number)]

    # 創建一個大小為 (number, row, column) 的三維列表，初始值為 0
    train_image = [[[0 for _ in range(column)] for _ in range(row)] for _ in range(number)]

    for i in range(number):
        train_label[i] = label_file.read(1)[0]
        for j in range(row):
            for k in range(column):
                train_image[i][j][k] = image_file.read(1)[0]

    label_file.close()
    image_file.close()

    label_file = open("C:\\Users\\yoush\\Desktop\\機器學習\\HW2\\t10k-labels.idx1-ubyte_", "rb")
    image_file = open("C:\\Users\\yoush\\Desktop\\機器學習\\HW2\\t10k-images.idx3-ubyte_", "rb")

    label_file.read(8)
    image_file.read(4)
    number = int.from_bytes(image_file.read(4), byteorder='big')
    row = int.from_bytes(image_file.read(4), byteorder='big')
    column = int.from_bytes(image_file.read(4), byteorder='big')

    # 創建一個大小為 number 的列表，初始值為 0
    test_label = [0 for _ in range(number)]
    # 創建一個大小為 (number, row, column) 的三維列表，初始值為 0
    test_image = [[[0 for _ in range(column)] for _ in range(row)] for _ in range(number)]
    
    for i in range(number):
        test_label[i] = label_file.read(1)[0]
        for j in range(row):
            for k in range(column):
                test_image[i][j][k] = image_file.read(1)[0]

    label_file.close()
    image_file.close()

    return train_label, train_image, test_label, test_image
    
train_label, train_image, test_label, test_image = load()


# %% train data中的圖片


rows = 300  
cols = 200  

# 每張圖片的高度和寬度
img_height = 28
img_width = 28

# 創建一個大的空白圖片，用來容納所有 60,000 張 28x28 的圖片
combined_image = [[0 for _ in range(cols * img_width)] for _ in range(rows * img_height)]

for idx in range(rows * cols):
    row_pos = (idx // cols) * img_height
    col_pos = (idx % cols) * img_width
    for j in range(img_height):
        for k in range(img_width):
            combined_image[row_pos + j][col_pos + k] = train_image[idx][j][k]

plt.figure(figsize=(20, 20))  
plt.imshow(combined_image, cmap='gray')
plt.axis('off')  
plt.show()


# %%

def find_max_value_index(list):
    max_index = 0  
    max_value = list[0] 
    for index in range(len(list)):
        if list[index] > max_value:
            max_value = list[index]
            max_index = index
    return max_index
# %%

def discrete_mode(train_label, train_image, test_label, test_image):
    prior = [0 for _ in range(10)]
    
    dim1 = 10  # 數字有10類(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    dim2 = 28  # 圖片長
    dim3 = 28  # 圖片寬
    dim4 = int(256/8) # 圖片像素0~255 數值每8個分成一組

    likelihood = [[[[1 for _ in range(dim4)] for _ in range(dim3)] for _ in range(dim2)] for _ in range(dim1)]

    # 計算prior和likelihood
    for i in range(len(train_label)):
        prior[train_label[i]] += 1
        for j in range(dim2):
            for k in range(dim3):
                id_1 = train_label[i]
                value = train_image[i][j][k]
                id_3 = int(value / 8)
                likelihood[id_1][j][k][id_3] += 1


    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                for l in range(dim4):
                    likelihood[i][j][k][l] = likelihood[i][j][k][l] / prior[i]

    prior = [x / sum(prior) for x in prior]   

    err = 0
    posterior = [[0 for _ in range(10)] for _ in range(len(test_label))]
    predictions = [0 for _ in range(len(test_label))]
    
    for i in range(len(test_label)):
        # 計算posterior = prior X likelihood => in log scale, posterior = log10(prior) + log10(likelihood)
        for j in range(10):    
            posterior[i][j] += math.log10(prior[j])
            for k in range(28):
                for l in range(28):
                    value = test_image[i][k][l]
                    id_3 = int(value / 8)
                    posterior[i][j] += math.log10(likelihood[j][k][l][id_3])

        # 找到後驗機率最大的類別
        predictions[i] = find_max_value_index(posterior[i])

        if predictions[i] != test_label[i]:
            err += 1

    err_rate = err / len(predictions)
    
    with open('discrete_output.txt', 'w') as f:   
        f.write("Posterior (in log scale):\n\n")
        for image_index in range(len(predictions)):
           f.write("Posterior probabilities:\n\n")
           for label in range(10):
                f.write(f"{label}: {posterior[i][label]/sum(posterior[i])}\n")
           f.write(f"Prediction: {predictions[i]}, Ans: {test_label[i]}\n\n")
           f.write("\n\n")
    
        
        f.write("Imagination of numbers in Bayesian classifier:\n\n")

        for label in range(10):
            f.write(f"Label {label}:\n\n")
            
            row_values = [[0 for _ in range(28)] for _ in range(28)]           
            for i in range(28):
                for j in range(28):
                    classifier_value = find_max_value_index(likelihood[label][i][j])
                    row_values[i][j] = int(classifier_value / 16)
                    #(題目要求)print a binary image which 0 represents a white pixel, and 1 represents a black pixel
                    #所以除以16
            for i in range(28):
                for j in range(28):
                    f.write(f"{row_values[i][j]} ")
                f.write("\n")  
            
            f.write("\n\n")  

        f.write(f"Error Rate: {err_rate}\n")      
        
        os.startfile('discrete_output.txt')



# %% 

def continuous_mode(train_label, train_image, test_label, test_image):
    # 初始化 prior 為每個數字 (0-9) 出現的次數
    prior = [0 for _ in range(10)]
    
    dim1 = 10  # 數字有 10 類 (0-9)
    dim2 = 28  # 圖片長
    dim3 = 28  # 圖片寬
    dim4 = 2   # mean and variance
    
    likelihood = [[[[1 for _ in range(dim4)] for _ in range(dim3)] for _ in range(dim2)] for _ in range(dim1)]
    
    # 計算 prior 和 mean
    for i in range(len(train_label)):
        label = train_label[i]
        prior[label] += 1
        for j in range(dim2):
            for k in range(dim3):
                likelihood[label][j][k][0] += train_image[i][j][k]
    

    
    # 計算 mean
    for label in range(10):
        for j in range(dim2):
            for k in range(dim3):
               likelihood[label][j][k][0] /= prior[label] 
    
    # 計算 variance
    for i in range(len(train_label)):
        label = train_label[i]
        for j in range(dim2):
            for k in range(dim3):
                value = train_image[i][j][k]
                likelihood[label][j][k][1] += (value - likelihood[label][j][k][0]) ** 2
    
    # 根據每個數字的出現次數，將 variance 進行平均
    for label in range(10):
        for j in range(dim2):
            for k in range(dim3):
                    likelihood[label][j][k][1] = likelihood[label][j][k][1] / prior[label]
    
    prior = [x / sum(prior) for x in prior]
    
    
    err = 0
    posterior = [[0 for _ in range(10)] for _ in range(len(test_label))]
    predictions = [0 for _ in range(len(test_label))] 

    with open('continuous_output.txt', 'w') as f:
        f.write("Posterior (in log scale):\n\n")
        for i in range(len(test_label)):
            # 計算 posterior = prior X likelihood => in log scale, posterior = log10(prior) + log10(likelihood)
            for j in range(10):    
                posterior[i][j] = math.log10(prior[j])  
                for k in range(dim2):
                    for l in range(dim3):
                        value = test_image[i][k][l]
                        mean = likelihood[j][k][l][0]
                        variance = likelihood[j][k][l][1]
                        try:
                            log_likelihood = math.log10(1 / (math.sqrt(2 * math.pi * variance))) - \
                                    ((value - mean) ** 2) / (2 * variance * math.log(10))

                            posterior[i][j] += log_likelihood
                        except ValueError:
                            # 如果出現數學錯誤，我們跳過這個特徵
                            continue
            
            predictions[i] = find_max_value_index(posterior[i])
            
            if predictions[i] != test_label[i]:
                err += 1
            f.write("Posterior probabilities:\n\n")
            for label in range(10):
                f.write(f"{label}: {posterior[i][label]/sum(posterior[i])}\n")
            f.write(f"Prediction: {predictions[i]}, Ans: {test_label[i]}\n\n")
            f.write("\n\n")
   
    err_rate = err / len(predictions)
   
    with open('continuous_output.txt', 'a') as f: 
        f.write("Imagination of numbers in Bayesian classifier:\n\n")

        for label in range(10):
            f.write(f"Label {label}:\n\n")
            

            row_values = [[0 for _ in range(28)] for _ in range(28)]
            
            for i in range(28):
                for j in range(28):
                    classifier_value = likelihood[label][i][j][0]
                    row_values[i][j] = int(classifier_value / 128)  
    
            for i in range(28):
                for j in range(28):
                    f.write(f"{row_values[i][j]} ")
                f.write("\n")  
            
            f.write("\n\n")  

        f.write(f"Error Rate: {err_rate}\n")  
        os.startfile('continuous_output.txt')


# %%

train_label, train_image, test_label, test_image = load()

def mode():
    x = int(input("input the mode( 0 : discrete, 1: continuous): " ))
    if x == 0:
        return discrete_mode(train_label, train_image, test_label, test_image)
    elif x == 1:
        return continuous_mode(train_label, train_image, test_label, test_image)

#%% x = 0

mode()

# %% x = 1

mode()


# %%2

def factorial(n):
    value = 1
    for i in range(n):
        value *= (i + 1)
    return value

def Bin(n, x):
    # n : the number of trials # x : the number of chance to see 1
    p = float( x / n )
    
    a1 = factorial(n)
    a2 = factorial(x)
    a3 = factorial(n-x)
    c = a1 / (a2 * a3)
    
    likelihood = c * (p**x) * ((1-p)**(n-x))
    
    return likelihood

def online_learning(path):
    a = int(input("input a (a >= 0) : "))
    b = int(input("input b (b >= 0) : "))
    c = 1
    for line in path.readlines():
        line = line.strip()
        print(f'case {c} : {line}\n')
        now_input  = list(line)
        for i in range(len(now_input)):
            now_input[i] = int(now_input[i])
            
        x = 0
        
        for i in range(len(now_input)):
            if now_input[i] == 1 :
                x = x + 1
     
        likelihood = Bin(len(now_input), x)
        print(f'Lilelihood: {likelihood}\n')
        print(f'Beta prior: a = {a} b= {b}\n')
        a = a + x
        b = len(now_input) + b - x
        print(f'Beta posterior: a = {a} b= {b}\n')
        print('\n')
        c += 1
    path.close


# %% case 1 a = 0, b = 0

path = open("C:\\Users\\yoush\\Desktop\\機器學習\\HW2\\testfile.txt.txt")
online_learning(path)

#%% case 2 a = 10 b = 1

path = open("C:\\Users\\yoush\\Desktop\\機器學習\\HW2\\testfile.txt.txt")
online_learning(path)

#%%3

def factorial(n):
    value = 1
    for i in range(n):
        value *= (i + 1)
    return value

def Bin(n, x, theta):
    a1 = factorial(n)
    a2 = factorial(x)
    a3 = factorial(n-x)
    c = a1 / (a2 * a3)
    
    likelihood = c * (theta**x) * ((1-theta)**(n-x))

    return likelihood

def beta_integral(a, b):
    delta = 1.0 / 10000
    total = 0.0
    for i in range(10000):
        theta = i * delta
        total += (theta**(a-1)) * ((1-theta)**(b-1)) * delta
    return total

def Beta(theta, a, b):
    
    c = beta_integral(a, b)
    
    prior = c * (theta**(a-1))* ((1-theta)**(b-1)) 
    
    return prior


def posterior(n, x, a, b, theta):
    likelihood = Bin(n, x, theta)
    prior = Beta(theta, a, b)
    return likelihood * prior

def trapezoidal_integral(x, y):
    total = 0.0
    for i in range(1, len(x)):
        total += 0.5 * (x[i] - x[i-1]) * (y[i] + y[i-1])
    return total
#未標準化的後驗概率是似然函數和先驗的乘積，這個值通常小於1。
#標準化後，我們得到的是真正的概率密度函數。對於集中的分布，峰值可能大於1，但整個函數在[0,1]區間的積分等於1。
def plot_posterior(n, x, a, b):
    theta_values = np.linspace(0, 1, 1000)
    posterior_values = [posterior(n, x, a, b, theta) for theta in theta_values]
    
    normalization_constant = trapezoidal_integral(theta_values, posterior_values)
    
    normalized_posterior = [p / normalization_constant for p in posterior_values]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(theta_values, posterior_values, '#1f77b4', linewidth=2)
    ax1.set_title(f'Unnormalized Posterior (n={n}, x={x}, a={a}, b={b})')
    ax1.set_xlabel('θ')
    ax1.set_ylabel('Unnormalized Posterior')
    ax1.grid(True, alpha=0.3)
    ax1.fill_between(theta_values, posterior_values, alpha=0.3, color='#aec7e8')
    
    ax2.plot(theta_values, normalized_posterior, '#d62728', linewidth=2)
    ax2.set_title(f'Normalized Posterior (n={n}, x={x}, a={a}, b={b})')
    ax2.set_xlabel('θ')
    ax2.set_ylabel('Posterior Probability Density')
    ax2.grid(True, alpha=0.3)
    ax2.fill_between(theta_values, normalized_posterior, alpha=0.3, color='#ff9896')
    
    plt.tight_layout()
    plt.show()


n, x, a, b = 10, 7, 2, 2
plot_posterior(n, x, a, b)
