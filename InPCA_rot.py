from scipy import *
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.optimize import minimize
    
def logDotProd(mu1,sigma1,mu2,sigma2):
    return log(2*sigma1*sigma2)/2-log(sigma1**2+sigma2**2)/2-1/4*(mu1-mu2)**2/(sigma1**2+sigma2**2)

def dotProd(mu1,sigma1,mu2,sigma2):
    return sqrt((2*sigma1*sigma2)/(sigma1**2+sigma2**2))*exp(-1/4*(mu1-mu2)**2/(sigma1**2+sigma2**2))

muRange = linspace(-4,4,50)
sigmaRange = 10**linspace(-4,4,50)
pairs = [[mu,sigma] for mu in muRange for sigma in sigmaRange]

W = [[logDotProd(mu1,sigma1,mu2,sigma2) for mu1,sigma1 in pairs] for mu2,sigma2 in pairs]

W_sum = [sum(row)/len(row) for row in W]
W_sum_sums = sum(W_sum)/len(W)
CovMat = [[W[i][j]-W_sum[i] - W_sum[j] + W_sum_sums for i in range(len(W))] for j in range(len(W))]

w,vt = LA.eig(CovMat)
metric_sig=sign(w.real)
print(w[0:8].real)
w = abs(w[0:8].real)
v = vt.T.real[0:8]
projMatEigs = dot(v.T,diag(sqrt(w)))

plt.plot(log(w/max(abs(w))))
# Most of the features are captured by the first 5 principal axes
# Approximately, our manifold lives in (3+2) Minkowski space.
# In (3+2) dimension space, (ds)**2=-(dx1)**2+(dx2)**2+(dx3)**2-(dx4)**2+(dx5)**2
# To have a better intepretation, one may perform isometric transformation on the manifold
# Isometric transformation means (ds')**2=(ds)**2
# This amounts to performing rotation or boost or translation to the manifold. 
# As we want to minimize the projection of manifold onto higher axes, we consider only rotation and boosts
# Rotation are performed on axes with the same metric signature (++/--)
# Boosts are performed on axes with different metric signatur (+-/-+)

# We shall try to perform transformation such that the 4th and 5th direction have very thin projections
# The proposed algorithm will perform boosts/rotation of 5th direction with respect to first 4 axes.
# Subsequently, rotation/boosts will be performed on the 4th direction with respect to the first 3 axes.

# Generally, we perform rotation/boosts on the nth axis with respect to the first n-1 axes. 
# After that, we perform similar operation on the n-1 th axis with respect to the first n-2 axes...

# Functions below are defined with the assumption that rows of matrices represents projection vectors on nth axis
def boosts(mat,axis1,axis2,phi):
    mat[axis1]=cosh(phi)*mat[axis1]+sinh(phi)*mat[axis2]
    mat[axis2]=cosh(phi)*mat[axis2]+sinh(phi)*mat[axis1]
    return mat

def rotation(mat,axis1,axis2,theta):
    mat[axis1]=cos(theta)*mat[axis1]-sin(theta)*mat[axis2]
    mat[axis2]=cos(theta)*mat[axis2]+sin(theta)*mat[axis1]
    return mat

def cost_func(mat,axis1,axis2,arg,n):
    if n==1:    
        mat=rotation(mat,axis1,axis2,arg)
    else:
        mat=boosts(mat,axis1,axis2,arg)
    return sum(mat[axis1]**2)     

def isometric_transformaton(mat,metric_sig,axis_min):
    axis1=axis_min
    for i in range(axis1):
        axis2=i
        # check the nature of the axes, if the metric is the same, we perform rotation else, boost
        if metric_sig[axis1]-metric_sig[axis2]==0:
            theta0=0;
            res=minimize(cost_func(mat,axis1,axis2,theta,1), theta0)
            # algo for minimization of manifold projection along nth direction
            # criteria: minimize the sum of projection**2 along nth direction
            theta=res.x
            mat=rotation(mat,axis1,axis2,theta)
        else:
                # algo for minimization of manifold projection along nth direction
                # criteria: minimize the sum of projection**2 along nth direction
            phi0=0;
            res=minimize(cost_func(mat,axis1,axis2,phi,0), phi0)
            phi=res.x
            mat=boosts(mat,axis1,axis2,phi)
    return mat
