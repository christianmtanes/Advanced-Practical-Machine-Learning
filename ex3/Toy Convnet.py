from scipy import signal
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

# define the functions we would like to predict:
num_of_functions = 3
size = 4
W = 4 * (np.random.random((size, size)) - 0.5)
y = {
    0: lambda x: np.sum(np.dot(x, W), axis=1),
    1: lambda x: np.max(x, axis=1),
    2: lambda x: np.log(np.sum(np.exp(np.dot(x, W)), axis=1))
}

def loss_func(p, y, lamb, w):
    return np.mean((p - y) ** 2) + 0.5 * lamb * np.linalg.norm(w) ** 2

def learn_linear(X, Y, batch_size, lamb, iterations, learning_rate):
    """
    learn a linear model for the given functions.
    :param X: the training and test input
    :param Y: the training and test labels
    :param batch_size: the batch size
    :param lamb: the regularization parameter
    :param iterations: the number of iterations
    :param learning_rate: the learning rate
    :return: a tuple of (w, training_loss, test_loss):
            w: the weights of the linear model
            training_loss: the training loss at each iteration
            test loss: the test loss at each iteration
    """

    training_loss = {func_id: [] for func_id in range(num_of_functions)}
    test_loss = {func_id: [] for func_id in range(num_of_functions)}
    w = {func_id: np.zeros(size) for func_id in range(num_of_functions)}

    for func_id in range(num_of_functions):
        for _ in range(iterations):

            # draw a random batch:
            idx = np.random.choice(len(Y[func_id]['train']), batch_size)
            x, y = X['train'][idx,:], Y[func_id]['train'][idx]

            # calculate the loss and derivatives:
            p = np.dot(x, w[func_id])
            loss = loss_func(p, y, lamb,w[func_id])
            p_test =  np.dot(X['test'], w[func_id])
            iteration_test_loss = loss_func(p_test, Y[func_id]['test'], lamb, w[func_id])
            dl_dw = (np.dot(2 * (p - y), x)) / batch_size + lamb * w[func_id]
            # update the model and record the loss:
            w[func_id] -= learning_rate * dl_dw
            training_loss[func_id].append(loss)
            test_loss[func_id].append(iteration_test_loss)

    return w, training_loss, test_loss


def forward(cnn_model, x):
    """
    Given the CNN model, fill up a dictionary with the forward pass values.
    :param cnn_model: the model
    :param x: the input of the CNN
    :return: a dictionary with the forward pass values
    """

    fwd = {}
    fwd['x'] = x
    fwd['o1'] = np.maximum(np.zeros(np.shape(x)), signal.convolve2d(x, [np.array(cnn_model['w1'])], mode='same'))
    fwd['o2'] = np.maximum(np.zeros(np.shape(x)), signal.convolve2d(x, [cnn_model['w2']], mode='same'))
    
    o1_12 = fwd['o1'][:,:2]
    o1_34 = fwd['o1'][:,2:]
    o2_12 = fwd['o2'][:,:2]
    o2_34 = fwd['o2'][:,2:]
    
    m1_1 = np.max(o1_12, axis=1)
    m2_1 = np.max(o1_34, axis=1)
    m1_2 = np.max(o2_12, axis=1)
    m2_2 = np.max(o2_34, axis=1)
    fwd['m'] = np.array([m1_1, m2_1,m1_2, m2_2]).T
    
    m1_1_arg_max = np.argmax(o1_12, axis=1)
    m2_1_arg_max = np.argmax(o1_34, axis=1) + 2
    m1_2_arg_max = np.argmax(o2_12, axis=1)
    m2_2_arg_max = np.argmax(o2_34, axis=1) + 2
    fwd['m_argmax'] = np.array([m1_1_arg_max, m2_1_arg_max, m1_2_arg_max, m2_2_arg_max]).T
    fwd['p'] = np.dot(fwd['m'], cnn_model['u'])
    return fwd


def backprop(model, y, fwd, batch_size):
    """
    given the forward pass values and the labels, calculate the derivatives
    using the back propagation algorithm.
    :param model: the model
    :param y: the labels
    :param fwd: the forward pass values
    :param batch_size: the batch size
    :return: a tuple of (dl_dw1, dl_dw2, dl_du)
            dl_dw1: the derivative of the w1 vector
            dl_dw2: the derivative of the w2 vector
            dl_du: the derivative of the u vector
    """
    dl_dp = 2 * (fwd['p'] - y) 
    dp_du = fwd['m']
    dl_du = np.dot(dl_dp.T, dp_du) / batch_size
    dp_dm = model['u'].T
    
    dl_dw1 = np.zeros((batch_size, 3))
    for i in range(batch_size):
        dm1_do1 = np.zeros((2, 4))
        dm1_do1[0,fwd['m_argmax'][i,0]] = 1
        dm1_do1[1,fwd['m_argmax'][i,1]] = 1
        x1 = fwd['x'][i,0]
        x2 = fwd['x'][i,1]
        x3 = fwd['x'][i,2]
        x4 = fwd['x'][i,3]
        do_dw_x = np.array([[0, x1, x2],
                            [x1, x2, x3],
                            [x2, x3, x4],
                            [x3, x4, 0]])
        do1_dw1 = do_dw_x.copy()
        for j in range(4):
            do1_dw1[j] *= bool(fwd['o1'][i,j])
        dl_dw1[i] = dl_dp[i] * dp_dm[:2].dot(dm1_do1).dot(do1_dw1)        
    
    dl_dw2 = np.zeros((batch_size, 3))
    for i in range(batch_size):
        dm2_do2 = np.zeros((2, 4))
        dm2_do2[0,fwd['m_argmax'][i,2]] = 1
        dm2_do2[1,fwd['m_argmax'][i,3]] = 1
        x1 = fwd['x'][i,0]
        x2 = fwd['x'][i,1]
        x3 = fwd['x'][i,2]
        x4 = fwd['x'][i,3]
        do_dw_x = np.array([[0, x1, x2],
                            [x1, x2, x3],
                            [x2, x3, x4],
                            [x3, x4, 0]])
        do2_dw2 = do_dw_x.copy()
        for j in range(4):
            do2_dw2[j] *= bool(fwd['o2'][i,j])
        dl_dw2[i] = dl_dp[i] * dp_dm[2:].dot(dm2_do2).dot(do2_dw2) 
    dl_dw1 = dl_dw1.mean(axis=0)
    dl_dw2 = dl_dw2.mean(axis=0)
    return (dl_dw1, dl_dw2, dl_du)


def learn_cnn(X, Y, batch_size, lamb, iterations, learning_rate):
    """
    learn a cnn model for the given functions.
    :param X: the training and test input
    :param Y: the training and test labels
    :param batch_size: the batch size
    :param lamb: the regularization parameter
    :param iterations: the number of iterations
    :param learning_rate: the learning rate
    :return: a tuple of (models, training_loss, test_loss):
            models: a model for every function (a dictionary for the parameters)
            training_loss: the training loss at each iteration
            test loss: the test loss at each iteration
    """

    training_loss = {func_id: [] for func_id in range(num_of_functions)}
    test_loss = {func_id: [] for func_id in range(num_of_functions)}
    models = {func_id: {} for func_id in range(num_of_functions)}

    for func_id in range(num_of_functions):

        # initialize the model:
        models[func_id]['w1'] = 5 * np.random.randn(3)
        models[func_id]['w2'] = 5 * np.random.randn(3) 
        models[func_id]['u'] = 5 * np.random.randn(4)

        # train the network:
        for _ in range(iterations):

            # draw a random batch:
            idx = np.random.choice(len(Y[func_id]['train']), batch_size)
            x, y = X['train'][idx,:], Y[func_id]['train'][idx]

            # calculate the loss and derivatives using back propagation:
            fwd = forward(models[func_id], x)
            param_vector = np.concatenate((models[func_id]['w1'],models[func_id]['w2'],models[func_id]['u']))
            loss = loss_func(fwd['p'], y, lamb, param_vector)
            dl_dw1, dl_dw2, dl_du = backprop(models[func_id], y, fwd, batch_size)

            # record the test loss before updating the model:
            test_fwd = forward(models[func_id], X['test'])
            iteration_test_loss = loss_func(test_fwd['p'], Y[func_id]['test'], lamb, param_vector)

            # update the model using the derivatives and record the loss:
            models[func_id]['w1'] -= learning_rate * dl_dw1
            models[func_id]['w2'] -= learning_rate * dl_dw2
            models[func_id]['u'] -= learning_rate * dl_du
            training_loss[func_id].append(loss)
            test_loss[func_id].append(iteration_test_loss)

    return models, training_loss, test_loss


if __name__ == '__main__':

    # generate the training and test data, adding some noise:
    X = dict(train=5 * (np.random.random((1000, size)) - .5),
             test=5 * (np.random.random((200, size)) - .5))
    Y = {i: {
        'train': y[i](X['train']) * (
        1 + np.random.randn(X['train'].shape[0]) * .01),
        'test': y[i](X['test']) * (
        1 + np.random.randn(X['test'].shape[0]) * .01)}
         for i in range(len(y))}
    linear_iterations = 2000
    w, training_loss, test_loss = learn_linear(X, Y, batch_size=32, lamb=0.1, iterations=linear_iterations, learning_rate=0.001)
    for i in range(num_of_functions):
        plt.figure()
        plt.plot(range(linear_iterations), training_loss[i], label='Train loss')
        plt.plot(range(linear_iterations), test_loss[i], label='Test loss')
        plt.title("learn_linear function y"+str(i))
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.savefig("learn_linear function y"+str(i))
    cnn_iterations = 5000
    models, training_loss, test_loss = learn_cnn(X, Y, batch_size=128, lamb=0.1, iterations=cnn_iterations, learning_rate=0.001)
    for i in range(num_of_functions):
        plt.figure()
        plt.plot(range(cnn_iterations), training_loss[i], label='Train loss')
        plt.plot(range(cnn_iterations), test_loss[i], label='Test loss')
        plt.title("learn_cnn function y"+str(i))
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.savefig("learn_cnn function y"+str(i))