import numpy as np
import scipy.linalg
import copy
import matplotlib.pyplot as plt



def Center_Data(A):
    """Function for creating covariances matrixes given data matrix

    Args:
        A (np.array): A data matrix

    Returns:
       sigma (np.array): The covariance matrix for the data matrix, A
       Aprime (np.array): The center matrix of A 

    """
    total = 0
    Aprime = copy.deepcopy(A)

    numRows = len(A)
    numCols = len(A[0])
    for i in range(numCols):
        for j in range(numRows):
            total += A[j][i]
        mean = total / numRows
        for j in range(numRows):
            Aprime[j][i] = A[j][i] - mean
        total = 0
        mean = 0
    inner = np.matmul(Aprime.T, Aprime)
    sigma = inner / (numRows - 1)
    return sigma, Aprime

def Calculate_PCs(sigma2):
    """Function for generating the eigenvector matrix and eigenvalue matrix using a QR decomposition.

    Args:
        sigma2 (np.array): A covariance matrix

    Returns:
        sigma (np.array): Matrix with the eigenvalues as its diagonal.
        eigenVectors (np.array): Matrix with the eigenvectors as the columns.
    """
    sigma = copy.deepcopy(sigma2)
    tol = 1e-6
    maxIters = 1000000

    lastDiag = -1
    currDiag = 0
    eigenVectors = np.identity(len(sigma))

    for iters in range(maxIters):

        lastDiag = currDiag
        currDiag = 0

        Q, R = scipy.linalg.qr(sigma, pivoting=False)

        eigenVectors = np.matmul(eigenVectors, Q)

        sigma = np.matmul(R, Q)
        for i in range(len(sigma)):
            currDiag = sigma[i][i]
        if ((abs(currDiag - lastDiag)) > tol):
            break
    return sigma, eigenVectors


if __name__ == "__main__":
    data = np.loadtxt("CleanedGaltonFamilies.csv", skiprows=1, delimiter=",")
    data = data[:, 1:]
    
    childHeight = np.loadtxt("ChildrensHeight.csv", skiprows=1, delimiter=",")
    childHeight = childHeight[:, 1:]
    
    
    covarianceMatrix, Aprime = Center_Data(data)
    
    eigenvalues, eigenvectors = Calculate_PCs(covarianceMatrix)
    
    
    eigenvectors = np.argsort(eigenvectors)
    
    eigenvectors = eigenvectors[:, :-4]
    
    principalComponents = np.matmul(Aprime, eigenvectors)
    
    
    xDesign = np.vstack([np.ones(len(principalComponents)), principalComponents.T]).T
    
    beta, _, _, _ = np.linalg.lstsq(xDesign, childHeight, rcond=None)
    
    
    yPred = xDesign @ beta  
    
    
    plt.scatter(principalComponents[:, 0], childHeight, label="Children Height points")
    plt.plot(principalComponents[:, 0], yPred, color="red", label="Fitted regression line")
    plt.xlabel('x')
    plt.ylabel('Height in Inches')
    plt.legend()
    plt.show()








