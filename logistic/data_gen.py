#creating imports#
import numpy as np


def exact(tspan,u0,R):
    """
        The exact solution to the logistic equation
    """
    A = u0/(u0-1)

    f = lambda A,R,t : (A*np.exp(R*t))/(A*np.exp(R*t)-1)
    output = f(A,R,tspan)
    return output


def create_branch_data(u0_min,u0_max,R_min,R_max,n_points=10):
    """
        generates the input data into the branch network
    """

    #creating list of points#
    u = np.linspace(u0_min,u0_max,n_points+1)
    r = np.linspace(R_min,R_max,n_points+1)

    U,R = np.meshgrid(u,r)

    U = U.flatten()
    R = R.flatten()

    X = np.vstack((U,R))
    
    return X.T

def create_trunk_target(X_branch,t0=0,t1=1,N=100):
    """
        creates the trunk network input
        and finds the target values
    """

    tspan = np.linspace(t0,t1,N+1)

    Y = []

    for i in range(0,len(X_branch),1):
        u0,R = X_branch[i,:]
        y = exact(tspan,u0,R)
        Y.append(y)
    
    Y = np.array(Y)
    return tspan,Y

def generate():
    u0_min = 0.1
    u0_max = 0.9

    R_min = 0
    R_max = 1

    X_branch = create_branch_data(u0_min,u0_max,R_min,R_max)
    x_trunk,Y = create_trunk_target(X_branch)

    x_trunk = np.reshape(x_trunk,(len(x_trunk),1))
    np.savez('train.npz',branch=X_branch,trunk=x_trunk,y=Y)


if __name__ == "__main__":
    generate()
