import torch
torch.manual_seed(42)


########################### BASICS ##########################################
# Tensors - specialized multi dimensional arrays for computational efficiency
# Rank 0 - scalar, rank1 -> 1D Tensor, rank2 -> Matrix and similarly more

# Loss is scalar - rank 0
# token embedding is rank 1
# conv maps are rank 2
# batch of conv embeddings is rank 3, RGB image

# 4D -> Batch of RGB images [number of images(batch), channel, height, width]
# 5D Tensor -> Video Data [number of video(batch), number of frames in video, channel, height, width]


############################## Import and Versions ###########################
print("Torch Version: ", torch.__version__)
if(torch.cuda.is_available()):
    print("GPU is available!")
    print(f"Using GPU: {torch.cuda.get_device_name()}")
else:
    print("GPU not available")

############################# Creating Tensors ################################

# 1. Empty Tensors (contains garbage values, it allocates space but does not changes what was already there)
print("Empty Tensor: \n", torch.empty(2,3))

# 2. Zero Tensor
print("Zero Tensor: \n", torch.zeros(2,3))

# 3. ones
print("One Tensor: \n", torch.ones(2,3))

# 4. Random
print("Random Tensor: \n", torch.rand(2,3))

# 5. custom tensors
print("Custom Tensor: \n", torch.tensor([[1,2,3],[4,5,6]]))

# 6. Arange -> gives tensor of values 0->10 (not included) with jump of 2
print("Using Arange: \n", torch.arange(0,10,2))

# 7. Linspace -> gives tensors equidistant
print("Using Linspace: \n", torch.linspace(0,10,2))

# 8. eye -> identity matrix
print("Using eye: \n", torch.eye(5))

# 9. full -> tensor with one value
print("Using full: \n", torch.full((3,3), 5))

############################ Shapes ##########################################
# 1. Check Shape
X = torch.full((2,3),1)
print(X)
print("Shape of X: ", X.shape)

# 2. Create Empty Tensor of Same Shape
print("Same Shape Empty Tensor: \n", torch.empty_like(X))

# 3. ones liek
print("Same Shape Ones Tensor: \n", torch.ones_like(X))

# 4. zeros like
print("Same Shape Zero Tensor: \n", torch.zeros_like(X))

# 5. rand like
# torch.rand_like(X) this fails since X is int tensor and rand gives floats between (0,1)

######################### Data Types ############################################
print("Data Type of X: ", X.dtype)

# 1. Assigning DataType, now it works
print("Same Shape Rand Tensor: \n", torch.rand_like(X, dtype=torch.float))

# 2. Changing Datatype directly
print("Changed Datatype of X: \n", X.to(torch.float32))

######################## Basic Operations #######################################
print("Original X: ", X)
print("Addition: \n", X + 2)
print("Multiplication: ", X * 3)
print("Division: \n", X / 3)
print("Integer Divide: \n", X // 3)
print("Modulo 4: \n", X % 4)
print("Power 3: \n", X ** 3)

###################### ELement wise Operations #################################
# Above same operations canbe done on A op B, and this will be elementwise in all cases


##################### Reduction Operations ####################################
print("Sum of Tensor X: \n", torch.sum(X))
print("Sum Along Columns of X: \n", torch.sum(X, dim=0))
print("Sum Along Rows of X: \n", torch.sum(X, dim=1))
print("Mean of X: \n", torch.mean(X.to(torch.float)))
print("Mean of X along cols: \n", torch.mean(X.to(torch.float), dim=0))
print("Mean of X along rows: \n", torch.mean(X.to(torch.float), dim=1))
# more are torch.argmax, torch.argmin, torch.var and can be computed along rows and cols.

########################### Matrix Operations ####################################
X = torch.randint(size=(2,3),low=1,high=10)
Y = torch.randint(size=(3,4),low=1,high=10)
print("X, Y: \n")
print(X)
print(Y)
print("Matrix Mult -> X@Y: \n", torch.matmul(X,Y))

# Dot Product
vector1 = torch.tensor([1,2,3])
vector2 = torch.tensor([1,2,3])
print("Dot Product vector1 . vector2: \n", torch.dot(vector1, vector2))

# Transpose
print("Transposed X:  \n", torch.transpose(X, dim0 = 0, dim1 = 1)) # swap dimension to enter here

# Random float matrix
print("Random Float Matrix: \n", torch.randint(size=(2,5), low = 1, high = 100, dtype=torch.float32))

# Determinant
print("Determinant of IDentity Matrix: \n", torch.det(torch.eye(4)))

# Inverse
print("Inverse of Identity Matrix: \n", torch.inverse(torch.eye(4)))


############################# Comparisons ########################
X = torch.tensor([[1,2,3],[4,5,6]])
Y = torch.tensor([[3,2,1],[6,5,4]])

# Comparisons return a boolena mask
print(X > Y)
print(X < Y)
print(X == Y)
print(X != Y)
print(X >= Y)
print(X <= Y)

########################### Special Functions ##############################
T = torch.randint(size=(3,4),low = 2, high = 12)
print("T: \n", T)

# more functions to use
print("Log : \n", torch.log(T))
print("Exp: \n", torch.exp(T))
print("Sqrt: \n", torch.sqrt(T))
print("Sigmoid: \n", torch.sigmoid(T))
print("Softmax across cols: \n", torch.softmax(T.to(torch.float16), dim=0))
print("Softmax across rows: \n", torch.softmax(T.to(torch.float16), dim=1))
print("RelU ed: \n", torch.relu(T.to(torch.float16)))

# m+n and save in m
print("X: \n", X)
print("Y: \n", Y)
print("X+Y: \n", X.add_(Y))

# Operation op + _ = implace. For eg add_, relu_, softmax_ etc.
Y = X
X[0][0] = -10
print("Copy Y: \n", Y)
# but if we make change to X, this will happen in Y also

# NEw Copy
Y = X.clone()
X[0][0] = 0
print(X)
print(Y)

# Moving to CUDA
print(torch.cuda.is_available())
device = torch.device('cuda')
print(f"Using Device: {device}")

# Tensors on GPU
print("GPU Tensor: \n", torch.randint(size=(2,3), low = 1, high = 10, device=device))
print("Converting Tensor")
print(" Y.device = ", Y.device)
Y = Y.to(device)
print(" Y.device = ", Y.device)

# Performance Difference in Both
size = 10000
matrix_cpu1 = torch.rand((size,size), dtype=torch.float)
matrix_cpu2 = torch.rand((size,size), dtype=torch.float)

print(f"CPU Matrix 1 - shape - {matrix_cpu1.shape} and device - {matrix_cpu1.device} ")
print(f"CPU Matrix 2 - shape - {matrix_cpu2.shape} and device - {matrix_cpu2.device} ")

import time 
start_time = time.time()
cpu_result = torch.matmul(matrix_cpu1, matrix_cpu2)
end_time = time.time()
print(f"CPU Execution Time: {(end_time - start_time):.3f} seconds")

# Move to gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
matrix_cpu1 = matrix_cpu1.to(device=device)
matrix_cpu2 = matrix_cpu2.to(device=device)
print(f"CPU Matrix 1 - shape - {matrix_cpu1.shape} and device - {matrix_cpu1.device} ")
print(f"CPU Matrix 2 - shape - {matrix_cpu2.shape} and device - {matrix_cpu2.device} ")

start_time = time.time()
gpu_result = torch.matmul(matrix_cpu1, matrix_cpu2)
end_time = time.time()
print(f"GPU Execution Time: {(end_time - start_time):.3f} seconds")

