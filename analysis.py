import numpy as np

def linear_fit(x, y):

    assert x.shape[0] == y.shape[0]

    xind = np.isfinite(x)
    yind = np.isfinite(y)
    X = x[xind * yind]
    Y = y[xind * yind]

    N = X.shape[0]
    Ex = X.sum()/N
    Ey = Y.sum()/N
    Exx = (X*X).sum()/N
    Exy = (X*Y).sum()/N
    
    m = (Exy - Ex*Ey) / (Exx - Ex*Ex)
    b = (Exx*Ey - Ex*Exy) / (Exx - Ex*Ex)

    return m, b
