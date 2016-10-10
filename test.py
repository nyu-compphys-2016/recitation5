import numpy as np
import matplotlib.pyplot as plt
import analysis as an
import time

def makeA(N):
    A = np.random.rand(N,N)
    return A

def makeb(N):
    b = np.zeros(N)
    b[-1] = 100
    return b

if __name__ == "__main__":

    N = 10
    A = makeA(N)
    b = makeb(N)

    N = np.array([400,800,1600,3200,6400,10000])
    t = np.empty((N.shape[0],4))

    for i in range(N.shape[0]):
        print(N[i])
        t1 = time.time()
        A = makeA(N[i])

        t2 = time.time()
        b = makeb(N[i])

        t3 = time.time()
        x = np.linalg.solve(A,b)
        t4 = time.time()

        np.linalg.eig(A)
        t5 = time.time()

        t[i,0] = t2-t1
        t[i,1] = t3-t2
        t[i,2] = t4-t3
        t[i,3] = t5-t4

    m1, b1 = an.linear_fit(np.log(N), np.log(t[:,0]))
    m2, b2 = an.linear_fit(np.log(N), np.log(t[:,1]))
    m3, b3 = an.linear_fit(np.log(N), np.log(t[:,2]))
    m4, b4 = an.linear_fit(np.log(N), np.log(t[:,3]))

    print("Convergence A:     ", m1)
    print("Convergence b:     ", m2)
    print("Convergence solve: ", m3)
    print("Convergence eig:   ", m4)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(N, t[:,0], 'k+', mew=4, ms=10)
    ax.plot(N, t[:,1], 'b+', mew=4, ms=10)
    ax.plot(N, t[:,2], 'r+', mew=4, ms=10)
    ax.plot(N, t[:,3], 'g+', mew=4, ms=10)

    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.show()
