from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

def kmeans(x, k=16, maxiter=50):
    h, w, col = x.shape
    x = np.reshape(x, (-1,col))
    mu = np.array([np.mean(xk, axis=0) for xk in np.split(x, k)])
    c = np.random.randint(k, size=h*w)
    it = 0
    while it < maxiter:
        it += 1
        c_prev = c
        c_k = np.zeros((h*w, k))
        for j in range(k):
            c_k[:, j] = np.linalg.norm(x - mu[j], ord=2, axis=1)
        c = np.argmin(c_k, axis=1)
        for j in range(k):
            if not np.any(c==j): continue
            mu[j] = (x.T @ (c==j).astype(int)) / np.sum((c == j).astype(int))
    x = np.round(mu[c], decimals=0)
    x = np.reshape(x, (h, w, col))
    return x.astype(int)

if __name__ == "__main__":
    A = imread('../data/peppers-large.tiff')
    A_copy = np.copy(A)
    A_compress = kmeans(A_copy, k=16)
    plt.imshow(A_compress)
    plt.show()
