import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse


def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i].

    Args:
        X - a NumPy matrix of n data points, each with d - 1 features

    Returns: X_augment, an (n, d) NumPy array with the added feature for each datapoint
    """
    column_of_ones = np.zeros([len(X), 1]) + 1
    return np.hstack((column_of_ones, X))

def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    n = X.shape[0]
    d = X.shape[1]
    k = theta.shape[0]
    tau = temp_parameter
    
    K = (theta @ X.T) / tau
    
    c_arr = []  # collect c values to maximize
    for i in range(n):
        x = X[i, :]
        prod_arr = (theta @ x) / tau
        c_arr.append(np.max(prod_arr))
        
    c_arr = np.array(c_arr)
    c_arr = np.tile(c_arr, (K.shape[0], 1))
    
    H = np.apply_along_axis(np.exp, 0, K - c_arr)
    sum_arr = np.apply_along_axis(np.sum, 0, H)[np.newaxis, :]  # converted to row vector
    H = H / sum_arr  # H has shape (k, n)

    return H

def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        J - the cost value (scalar)
    """
    n, d = X.shape
    k = theta.shape[0]

    # Compute probabilities matrix H
    H = compute_probabilities(X, theta, temp_parameter)
    
    # Apply natural logarithm to all elements of matrix H
    H_2 = np.apply_along_axis(np.log, axis=0, arr=H)
    
    # Create a filter matrix Y_2 such that if x[i] belongs to label j, Y_2[i, j] = 1, and 0 otherwise.
    Y_2 = np.zeros((k, n))
    for i in range(n):
        Y_2[Y[i], i] = 1
        
    """
    Y_2 == 1 creates a boolean array of the same shape as Y_2, where each element is True if the corresponding element in Y_2 is 1 and False otherwise.
    H_2[Y_2 == 1] uses this boolean array to index H_2, returning a flattened array of elements where the boolean array is True.
    """
    val_sum = np.sum(H_2[Y_2 == 1])  # we get the values sum after boolean indexing.
    reg = lambda_factor/2 * np.ravel(theta**2).sum()  # regularization term
    
    J = -1/n * val_sum + reg
    return J

def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    n, d = X.shape
    k = theta.shape[0]
    H = compute_probabilities(X, theta, temp_parameter)  # compute probabilities matrix H
    gradients_matrix = np.zeros((k, d))
    # iterate over rows of theta matrix, where theta[m, :] is a vector
    # representing the parameters of our model for label/class m
    for m in range(k):
        y_bool = (Y == m).astype(int).reshape((n, 1))  # create a boolean array, reshape to row vector
        # each gradient will be a (1, d) row vector
        gradient = (-1/(temp_parameter*n) * np.sum(X * (y_bool.reshape((n, 1)) - H[m, :].reshape((n, 1))), axis=0) + lambda_factor*theta[m, :]).reshape((1, d))
        # vertically stack the gradient vectors as rows, will result in a (k, d) gradients matrix
        gradients_matrix[m, :] = gradient
    theta = theta - alpha * gradients_matrix
    return theta

def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each datapoint in the test set
    """
    return (np.mod(train_y, 3), np.mod(test_y, 3))

def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit. (mod 3)

    Args:
        X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    error_count = 0.
    assigned_labels = np.mod(get_classification(X, theta, temp_parameter), 3)
    return 1 - np.mean(assigned_labels == np.mod(Y, 3))

def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for i in range(num_iterations):
        cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
    return theta, cost_function_progression

def get_classification(X, theta, temp_parameter):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        theta - (k, d) NumPy array where row j represents the parameters of our model for
                label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis = 0)

def plot_cost_function_over_time(cost_function_history):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def compute_test_error(X, Y, theta, temp_parameter):
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)
