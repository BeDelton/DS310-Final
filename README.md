# DS310-Final
## QR Iterative Algorithm for Dimensional Reduction
The goal of this paper was to write a linear regression on how parents height and other factors predict a child's height. This was done using a QR Iterative algorithm to calculate the principal components of a data set in order to reduce the relationship between the variables we were examining. 
We use a QR Decomposition compared to a Power Iteration because we are interested in more than one of the eigenvalues. A Power Iteration converges at the most dominant eigenvalue and would need deflation to calculate the others, whereas a QR Decomposition will give multiple prominent eigenvalues. There are drawbacks with a QR Decomposition in that it is still highly costly and converges slowly.
