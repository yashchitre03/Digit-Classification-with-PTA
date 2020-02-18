import struct as st
import numpy as np
import matplotlib.pyplot as plt
import copy as cp


def read_input(file):
    """

    Args:
        file (idx): binary input file.

    Returns:
        numpy: arrays for our dataset.

    """
    
    with open(file, 'rb') as file:
        z, d_type, d = st.unpack('>HBB', file.read(4))
        shape = tuple(st.unpack('>I', file.read(4))[0] for d in range(d))
        return np.frombuffer(file.read(), dtype=np.uint8).reshape(shape)


def step_activation(local_induced_field):
    """
    
    Args:
        local_induced_field (numpy array): W*Xi.

    Returns:
        temp (numpy array): result of step activation.

    """
    
    temp = np.zeros((10, 1))
    for index, value in enumerate(local_induced_field):
        if value >= 0:
            temp[index] = 1
    return temp


def initialize_w():
    """
    
    Returns:
        initial_w (numpy array): initialized numpy array.

    """
    
    initial_w = np.random.uniform(-1, 1, size=(10, 784))
    return initial_w


def pta(n, eta, epsilon, temp_W):
    """

    Args:
        n (int): size of training samples.
        eta (int): learning rate.
        epsilon (double): threshold.
        temp_W (numpy array): initialized weights.

    Returns:
        numpy array: updated weights.

    """
    
    W = cp.deepcopy(temp_W)
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
            W = W + eta*np.dot((d_X - u_WX), X.T)
        if errors[epoch-1] / n <= epsilon:
            break
        elif epoch >= 50:
            plt.plot(range(epoch), errors)
            plt.xlabel("Number of epochs")
            plt.ylabel("Number of misclassifications")
            plt.show()
            return -1
        
    plt.plot(range(epoch), errors)
    plt.xlabel("Number of epochs")
    plt.ylabel("Number of misclassifications")
    plt.show()
    return W


def test_classification(W, n):
    """

    Args:
        W (numpy array): updated weights.
        n (int): size of training samples.

    Returns:
        None.

    """
    
    test_errors = 0
    no_rows = np.size(test_X, 0)
    for i in range(no_rows):
        if np.dot(W, test_X[i].reshape((784, 1))).argmax() != test_y[i]:
            test_errors += 1
    print(f"Percentage of misclassified test samples are (for training size -> {n}): {(test_errors/no_rows)*100}")


if __name__ == "__main__":
    """
    
    The methods to read the binary input and run the PTA algorithm starts below
    for different values of training size, and threshold.
    
    """
    train_X = read_input('train-images.idx3-ubyte')
    train_y = read_input('train-labels.idx1-ubyte')
    test_X = read_input('t10k-images.idx3-ubyte')
    test_y = read_input('t10k-labels.idx1-ubyte')
    initial_W = initialize_w()
    W = pta(50, 1, 0, initial_W)
    test_classification(W, 50)
    W = pta(1000, 1, 0, initial_W)
    test_classification(W, 1000)
    W = pta(60000, 1, 0, initial_W)
    if W != -1:
        test_classification(W, 60000)
    else:
        print("Algorithm failed to converge for training size -> 60000 and epsilon -> 0")
    W = pta(60000, 1, 0.15, initialize_w())
    test_classification(W, 60000)
    W = pta(60000, 1, 0.15, initialize_w())
    test_classification(W, 60000)
    W = pta(60000, 1, 0.15, initialize_w())
    test_classification(W, 60000)