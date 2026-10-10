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