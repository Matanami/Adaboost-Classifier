from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data
np.random.seed(7)


def run_adaboost(X_train, y_train, X_test, y_test, T):

    n = X_train.shape[0]
    D_array = np.zeros((T, n))
    h_array = []
    alpha_vals = []
    errors_train = []
    errors_test = []
    D_array[0] = np.full((n,), 1 / n)
    test_predict = np.zeros(X_test.shape[0])
    s_train_predict = np.zeros(X_train.shape[0])

    for t in range(T-1):
        h_theta, h_pred, error, h_index = WL(D_array[t], X_train, y_train)
        y_pred = classify(X_train, h_theta, h_pred, h_index)

        error = calculate_zero_one_error(y_pred, y_train, D_array[t])
        W_t = 0.5 * np.log((1 - error) / error)
        D_array[t + 1] = D_array[t] * np.exp(-W_t * y_train * y_pred) / np.sum(D_array[t] * np.exp(-W_t * y_train * y_pred))

        h_array.append((h_pred, h_index, h_theta))
        alpha_vals.append(W_t)

        s_train_predict += W_t * classify(X_train, h_theta, h_pred, h_index)
        train_error = calculate_zero_one_error(np.sign(s_train_predict), y_train)
        errors_train.append(train_error)

        test_predict += W_t * classify(X_test, h_theta, h_pred, h_index)
        test_error = calculate_zero_one_error(np.sign(test_predict), y_test)
        errors_test.append(test_error)

    return h_array, alpha_vals, errors_train, errors_test

def WL(D_t, X_train, y_train):
    best_error = float('inf')
    best_polarity = 1
    best_threshold = None
    index = None
    for feature_idx in range(X_train.shape[1]):
        thresholds = np.unique(X_train[:, feature_idx])
        X_feature = X_train[:, feature_idx]
        for threshold in thresholds:
            for polarity in [-1, 1]:
                predictions = polarity * np.where(X_feature <= threshold, 1, -1)
                error = calculate_zero_one_error(predictions, y_train, D_t)

                if error < best_error:
                    index = feature_idx
                    best_error = error
                    best_threshold = threshold
                    best_polarity = polarity

    return best_threshold, best_polarity, best_error, index


def calculate_zero_one_error(predictions, labels, weights=None):
    weights = np.ones(len(predictions)) if weights is None else weights
    incorrect = predictions != labels
    error = np.dot(weights, incorrect) / np.sum(weights)
    return error

def classify(X, threshold, polarity, index):
    X_feature = X[:, index]
    return polarity * np.where(X_feature <= threshold, 1, -1)

def strong_classify(X, hypotheses, alpha_vals):
    predict = np.zeros(X.shape[0])
    for t in range(len(hypotheses)):
        polarity, index, theta =hypotheses[t]
        X_feature = X[:,index]
        predict += alpha_vals[t] * polarity * np.where(X_feature <= theta, 1, -1)
    return predict



def main():
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data
    T = 80
    hypotheses, alpha_vals, errors_train, errors_test = run_adaboost(X_train, y_train, X_test, y_test, T)
    #a(errors_train, errors_test)
    c(hypotheses, alpha_vals, X_train, y_train, X_test, y_test,T)


def a(errors_train, errors_test):
    plt.plot(errors_train, label='Train Error')
    plt.plot(errors_test, label='Test Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig("a.png")
    plt.show()


def c(hypotheses, alpha_vals, X_train, y_train, X_test, y_test,T):
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    errors_train = []
    errors_test = []

    for t in range(1,T):

        train_pred = strong_classify(X_train, hypotheses[:t], alpha_vals[:t])
        train_loss = np.sum(np.exp(-y_train * train_pred))/train_size
        errors_train.append(train_loss)

        test_pred = strong_classify(X_test, hypotheses[:t], alpha_vals[:t])
        test_loss = np.sum(np.exp(-y_test * test_pred))/test_size
        errors_test.append(test_loss)

    plt.plot(errors_train, label='Train Error')
    plt.plot(errors_test, label='Test Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig("c.png")
    plt.show()


if __name__ == '__main__':
    main()
