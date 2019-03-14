import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

######### Load the data ##########

infile = open('faces.csv','r')
img_data = infile.read().strip().split('\n')
img = [map(int,a.strip().split(',')) for a in img_data]
pixels = []
for p in img:
    pixels += p
faces = np.reshape(pixels,(400,4096))   #each element is a 1D 64*64 nparray
######### Global Variable ##########

image_count = 0

######### Function that normalizes a vector x (i.e. |x|=1 ) #########

# > numpy.linalg.norm(x, ord=None, axis=None, keepdims=False) 
#   This function is able to return one of eight different matrix norms, 
#   or one of an infinite number of vector norms (described below), 
#   depending on the value of the ord parameter.
#   Note: in the given functionm, U should be a vector, not a array. 
#         You can write your own normalize function for normalizing 
#         the colomns of an array.

def normalize(U):
	return U / LA.norm(U)


######### Display first face #########

# Useful functions:
# > numpy.reshape(a, newshape, order='C')
#   Gives a new shape to an array without changing its data.
# > matplotlib.pyplot.figure()
# 	Creates a new figure.
# > matplotlib.pyplot.title()
#	Set a title of the current axes.
# > matplotlib.pyplot.imshow()
#	Display an image on the axes.
#	Note: You need a matplotlib.pyplot.show() at the end to display all the figures.

first_face = np.reshape(faces[0],(64,64),order='F')
image_count+=1
plt.figure(image_count)
plt.title('First_face')
plt.imshow(first_face,cmap=plt.cm.gray)

########## display a random face ###########

# Useful functions:
# > numpy.random.choice(a, size=None, replace=True, p=None)
#   Generates a random sample from a given 1-D array
# > numpy.ndarray.shape()
#   Tuple of array dimensions.
#   Note: There are two ways to order the elements in an array: 
#         column-major order and row-major order. In np.reshape(), 
#         you can switch the order by order='C' for row-major(default), 
#         or by order='F' for column-major. 


#### Your Code Here ####
a = np.array([i for i in range(400)])
random_index = np.random.choice(a)
random_face = np.reshape(faces[random_index],(64,64),order='F')
image_count += 1
plt.figure(image_count)
plt.title('Random_face')
plt.imshow(random_face,cmap=plt.cm.gray)

########## compute and display the mean face ###########

# Useful functions:
# > numpy.mean(a, axis='None', ...)
#   Compute the arithmetic mean along the specified axis.
#   Returns the average of the array elements. The average is taken over 
#   the flattened array by default, otherwise over the specified axis. 
#   float64 intermediate and return values are used for integer inputs.

#### Your Code Here ####
mean = np.mean(faces, axis=0)
image_count += 1
mean = np.reshape(mean, (64,64), order='F')
plt.figure(image_count)
plt.title('Mean_face')
plt.imshow(mean,cmap=plt.cm.gray)

######### substract the mean from the face images and get the centralized data matrix A ###########

# Useful functions:
# > numpy.repeat(a, repeats, axis=None)
#   Repeat elements of an array.

#### Your Code Here ####

mean = np.mean(faces, axis=0)
mean_1 = np.repeat(mean, 400, axis=None)
mean_2 = np.reshape(mean_1, (400,4096), 1)

A = faces - mean_2
'''
A_face = np.reshape(A[0],(64,64),order='F')
image_count+=1
plt.figure(image_count)
plt.title('A_face')
plt.imshow(A_face,cmap=plt.cm.gray)
'''

######### calculate the eigenvalues and eigenvectors of the covariance matrix #####################

# Useful functions:
# > numpy.matrix()
#   Returns a matrix from an array-like object, or from a string of data. 
#   A matrix is a specialized 2-D array that retains its 2-D nature through operations. 
#   It has certain special operators, such as * (matrix multiplication) and ** (matrix power).

# > numpy.matrix.transpose(*axes)
#   Returns a view of the array with axes transposed.

# > numpy.linalg.eig(a)[source]
#   Compute the eigenvalues and right eigenvectors of a square array.
#   The eigenvalues, each repeated according to its multiplicity. 
#   The eigenvalues are not necessarily ordered. 

#### Your Code Here ####
A_t = np.matrix.transpose(A)
L = np.matmul(A, A_t)
eigenvalues_L, eigenvectors_L = np.linalg.eig(L)

eigenvalues_V = eigenvalues_L
eigenvectors_V = np.empty([0, 4096])
for v in eigenvectors_L:
    new_v = np.matmul(A_t, np.array([v]).T)
    eigenvectors_V = np.append(eigenvectors_V, normalize(new_v.T), axis=0)

########## Display the first 16 principal components ##################

#### Your Code Here ####
for count in range(16):
    principle_comp = np.reshape(eigenvectors_V[count],(64,64),order='F')
    image_count+=1
    plt.figure(image_count)
    plt.title('Principle component number: %d'%(count+1))
    plt.imshow(principle_comp,cmap=plt.cm.gray)

########## Reconstruct the first face using the first two PCs #########

#### Your Code Here ####
new_first_face = np.dot(eigenvectors_V[0], faces[0])*eigenvectors_V[0] + \
                    np.dot(eigenvectors_V[1], faces[0])*eigenvectors_V[1] + \
                    np.mean(faces, axis=0)
                    
new_first_face = np.reshape(new_first_face,(64,64),order='F')
image_count += 1
plt.figure(image_count)
plt.title('First face constructed by the first two principal components')
plt.imshow(new_first_face,cmap=plt.cm.gray)

########## Reconstruct random face using the first 5, 10, 25, 50, 100, 200, 300, 399  PCs ###########

#### Your Code Here ####
new_face_100 =  np.dot(eigenvectors_V[4], faces[99])*eigenvectors_V[4] + \
                np.dot(eigenvectors_V[9], faces[99])*eigenvectors_V[9] + \
                np.dot(eigenvectors_V[24], faces[99])*eigenvectors_V[24] + \
                np.dot(eigenvectors_V[49], faces[99])*eigenvectors_V[49] + \
                np.dot(eigenvectors_V[99], faces[99])*eigenvectors_V[99] + \
                np.dot(eigenvectors_V[199], faces[99])*eigenvectors_V[199] + \
                np.dot(eigenvectors_V[299], faces[99])*eigenvectors_V[299] + \
                np.dot(eigenvectors_V[398], faces[99])*eigenvectors_V[398] + \
                np.mean(faces, axis=0)
                
new_face_100 = np.reshape(new_face_100,(64,64),order='F')
image_count += 1
plt.figure(image_count)
plt.title('The 100th face constructed by PCs')
plt.imshow(new_face_100,cmap=plt.cm.gray)

######### Plot proportion of variance of all the PCs ###############

# Useful functions:
# > matplotlib.pyplot.plot(*args, **kwargs)
#   Plot lines and/or markers to the Axes. 
# > matplotlib.pyplot.show(*args, **kw)
#   Display a figure. 
#   When running in ipython with its pylab mode, 
#   display all figures and return to the ipython prompt.

#### Your Code Here ####
image_count += 1
PC_sum = sum(eigenvalues_V)

portion = [i/PC_sum for i in eigenvalues_V]
X = [i+1 for i in range(len(eigenvalues_V))]

plt.figure(image_count)
plt.plot(X, portion)
plt.title('Proportion of variance explained by all the principal components\n')
plt.xlabel('Principal components')
plt.ylabel('Variance')






