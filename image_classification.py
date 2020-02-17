import struct as st
import numpy as np

np.random.seed(100)

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = st.unpack('>HBB', f.read(4))
        shape = tuple(st.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def step_activation(local_induced_field):
    temp = np.zeros((10, 1))
    for index, value in enumerate(local_induced_field):
        if value >= 0:
            temp[index] = 1
    return temp


def pta():
    
    learning_rate = 0.5
    threshold = 0.15
    n = 60000
    W = np.random.uniform(-1, 1, size=(10, 784))
    epoch = 0
    errors = []
    
    while True:
        errors.append(0)
        for i in range(n):
            if np.dot(W, train_X[i].reshape((784, 1))).argmax() != train_y[i]:
                errors[epoch] += 1
        epoch += 1
        for i in range(n):
            X = train_X[i].reshape((784, 1))
            d_X = np.eye(10, 1, train_y[i]*-1)
            WX = np.dot(W, X)
            u_WX = step_activation(WX.reshape((10, )))
            W = W + learning_rate*np.dot((d_X - u_WX), X.T)
        if errors[epoch-1] / n <= threshold:
            break
    return W


def test_classification(W):
    
    test_errors = 0
    for i in range(np.size(test_X, 0)):
        if np.dot(W, test_X[i].reshape((784, 1))).argmax() != test_y[i]:
            test_errors += 1
    print(test_errors)


if __name__ == "__main__":
    train_X = read_idx('train-images.idx3-ubyte')
    train_y = read_idx('train-labels.idx1-ubyte')
    test_X = read_idx('t10k-images.idx3-ubyte')
    test_y = read_idx('t10k-labels.idx1-ubyte')
    W = pta()
    test_classification(W)