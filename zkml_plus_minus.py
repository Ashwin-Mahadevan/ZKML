import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from gmpy2 import mpz


def threshold_array(arr, scale_to=10):
    ''' 
    The parameters here play the role of the
    learning rate in some sense.
    arr is an np array.
    scales the array into a specified range
    '''
    max_entry = np.max(arr)
    min_entry = np.min(arr)
    abs_max = max(abs(max_entry),abs(min_entry))
    res = arr.copy()
    res = res/abs_max
    res = res*scale_to
    res = res.astype(np.float64)
    res = np.around(res)
    res = res.astype(int)
    return res


class Model:
    def __init__(self, max_input, max_weight, structure):
        """
        max_input is an integer.
        max weight is some small int
        structure is a list that indicates the lengths 
        of layers from the input to output (discluding biases).
        Defines a model
        """
        self.max_input = max_input
        self.max_weight = max_weight
        self.structure= np.array(structure)
        self.depth = len(self.structure)-1  # #transition matrices and vectors
        self.weights = self.random_weights()
        self.biases = self.random_biases()
        self.max_for_layer = self.layer_max_vals()
        self.beta = self.max_for_layer[-1]
        
    def layer_max_vals(self):
        res = [mpz(self.max_input)]
        for n in self.structure[:-1]:
            layer_max = ((res[-1]*n) + 1)*self.max_weight
            res.append(layer_max)
        return res
        
    def predict(self, inp):
        x=inp.copy()
        if len(x.shape)==1:
            for i in range(self.depth):
                x = (
                    self.weights[i].dot(x).reshape(-1) + 
                    self.biases[i].reshape(-1)
                    )
        else:
            for mat, vec in zip(self.weights, self.biases):
                x = np.dot(mat, x) + vec
        return x

    def random_weights(self):
        '''returns random weights between 1 and max_weight'''
        res = []
        n = self.structure[0]
        res.append(np.random.randint(
            low = -self.max_weight-1,
            high = self.max_weight+1,
            size = (self.structure[1],n)
            ))
        for i in range(self.depth - 1):
            res.append(np.random.randint(
                low = -self.max_weight-1,
                high = self.max_weight+1,
                size = (self.structure[i+2],self.structure[i+1])
                ))
        return res
        
    def random_biases(self):
        '''returns random biases between 1 and max_weight'''
        res = []
        res.append(
            np.array(
                [[mpz(
                    np.random.randint(
                        -self.max_weight-1,
                        self.max_weight+1
                        )
                    )] for i in range(self.structure[1])]
                )
            )
        for i in range(self.depth - 1):
            res.append(
                np.array(
                    [[mpz(
                        np.random.randint(
                            -self.max_weight-1,
                            self.max_weight+1
                            )
                        )] for i in range(self.structure[i+2])]
                    )
                )
        return res
 
    def compute_layers(self, data):
        ''' returns a list of matrices with values in
        each layer after forward propagation
        '''
        res = []
        a = data.copy()
        for mat, vec in zip(self.weights, self.biases):
            a = np.dot(mat, a) + vec
            res.append(a)
        return res
    
    def compute_deltas(self, layers, lables):
        ''' returns deltas
        '''
        res = []
        for l in reversed(range(len(layers))):
            if l == len(layers)-1:
                d = layers[l] - lables
            else:
                d = self.weights[l+1].T.dot(d)
            res.append(d)
        res.reverse()
        return res
    
    def compute_gradients(self, layers, deltas):
        '''returns gradients for weights and biases.
        The first layer in layes is the input.
        '''
        gradW = []
        gradB = []
        for i in range(self.depth):
            # for weights:
            gw = layers[i] @ deltas[i].T
            gradW.append(gw)
            # for biases:
            gb = np.sum(deltas[i], axis = 1)
            gradB.append(gb)
        return gradW, gradB
    
    def fit(self,X,y,iterations, verbose=1):
        '''X is a database. y is a list of labels. 
        iterations is an integer. 
        '''
        for i in range(iterations):
            if verbose:
                print(f"Running interation {i+1} / {iterations}.")
            A = self.compute_layers(X)
            d = self.compute_deltas(A,y.T)
            Dw, Db = self.compute_gradients([X] + A, d)
            for i in range(self.depth):
                self.weights[i] -=  threshold_array(Dw[i]).T
                self.weights[i][self.weights[i]<-self.max_weight] = -self.max_weight
                self.weights[i][self.weights[i]>self.max_weight] = self.max_weight
                self.biases[i] -=  threshold_array(Db[i]).reshape(M.biases[i].shape)
                self.biases[i][self.biases[i]<-self.max_weight] = -self.max_weight
                self.biases[i][self.biases[i]>self.max_weight] = self.max_weight
        return
    
    def cost(self, data, labels):
        ''' data is a database feedable to self.predict
        lables is a list of wanted outputs as binary vectors.
        returns a float
        '''
        # compute predictions
        Y = self.predict(data)
        # set normalization factor
        # compute average square of differences
        return (np.sum(((Y.T) - labels)**2)/(2*len(data.T)))/(self.beta**2)


# defining the model 
print("Initializing model.")   
M = Model(255,1000, [784,16,16,10])


# loading mnist data
print("Loading MNIST dataset and arranging data.")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# changing orientation to run predict over entire database at once:
# (to predict a single sample use:  x_train.T[sample_index])
x_train = x_train.reshape(60000,784).transpose()
x_test = x_test.reshape(10000,784).transpose()


# Creating binary vectors for cost computation. 
# Beware! orientation is opposite to predict's output
y_train_bin = np.ones((60000,10), dtype = mpz)*(-M.beta)
for val, vec in zip(y_train, y_train_bin):
    vec[val] = M.beta # //M.normalizer
y_test_bin = np.ones((10000,10), dtype = mpz)*(-M.beta)
for val, vec in zip(y_test, y_test_bin):
    vec[val] = M.beta # //M.normalizer


#train model.
regular_training = 0
if regular_training:
    print("Training.")
    M.fit(x_train, y_train_bin, 3)


batch_training = 1
if batch_training:
    # split data to minibatches
    batch_size = 10000
    num_batches = 60000//batch_size
    x_mini_batches = [
        x_train.T[l*batch_size:(l+1)*batch_size].T for l in range(num_batches)
        ]
    y_mini_batches = [
        y_train_bin[l*batch_size:(l+1)*batch_size] for l in range(num_batches)
        ]
    
    # fit and plot running cost
    epochs = 5
    print("Training model.")
    running_cost = [M.cost(x_test, y_test_bin)]
    for epoch in range(epochs):
        for batch in range(num_batches):
            M.fit(
                x_mini_batches[batch],
                y_mini_batches[batch],
                1,
                verbose=1
                )
        c = M.cost(x_test,y_test_bin)
        running_cost.append(c)  
        print(f"Epoch {epoch+1} out of {epochs}\tCost: {c}")
    plt.plot(running_cost)

    print("Done training.\nAccuracy:")
    pred = M.predict(x_test).T
    predictions =[np.argmax(pred[i]) for i in range(10000)]
    match = y_test == predictions
    print(f"{sum(match)} out of 10,000.")
