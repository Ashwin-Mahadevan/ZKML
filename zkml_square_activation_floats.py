import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from gmpy2 import mpz


def fix_gradients(DW, DB, scaling = 1):
    W = [w.copy() for w in DW]
    B = [b.copy() for b in DB]
    # # remove outliers
    # for i in range(len(W)):
    #     W[i] = remove_outliers(W[i])
    #     B[i] = remove_outliers(B[i])
    
    # log
    # for i in range(len(W)):
    #     W[i] = my_log(W[i])
    #     B[i] = my_log(B[i])
        
    # scale to learning rate
    c = iteration_scaler(W, B)
    for i in range(len(W)): 
        W[i] = scale(W[i], scaler = c, scale_to = scaling)
        B[i] = scale(B[i], scaler = c, scale_to = scaling)
    return W, B

def my_log(mat):
    res = mat.copy()
    signs = np.sign(res)
    res = np.abs(res)
    res[res < 1] = 1
    res = np.log(np.array(res, dtype = float))
    res = res*signs
    return res

def remove_outliers(arr, low = 25, high = 75, threshold = 1.5):
    data = arr.copy()
    q1, q3 = np.percentile(data, [low, high])
    iqr = q3 - q1
    mad = np.median(np.abs(data - np.median(data)))
    data[
        (data < q1 - threshold*iqr) | (data > q3 + threshold*iqr)
        ] = np.median(data) + mad
    return data

def iteration_scaler(DW,DB):
    DB_max_vals = [abs_max(D) for D in DB]
    DW_max_vals = [abs_max(D) for D in DW]
    maxes = DW_max_vals + DB_max_vals
    return max(maxes)    

def abs_max(arr):
    max_entry = np.max(arr)
    min_entry = np.min(arr)
    return max(abs(max_entry),abs(min_entry))

def scale(arr, scaler, scale_to=1):
    ''' 
    The parameters here play the role of the
    learning rate in some sense.
    arr is an np array.
    scales the array into a specified range
    '''
    res = arr.copy()
    res = res/scaler
    res = res*scale_to
    res = res.astype(np.float64)
    res = np.around(res)
    return res


class Model:
    def __init__(
            self,
            max_input = 255,
            max_weight = 10000,
            structure = [784, 16, 16, 10],
            weights_scaler = 100,
            biases_scaler = 500,
            learning_rate = 2
            ):
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
        self.weights = self.random_weights(weights_scaler)
        self.biases = self.random_biases(biases_scaler)
        self.max_for_layer = self.layer_max_vals()
        self.beta = self.max_for_layer[-1]
        self.learning_rate = learning_rate
        
    def layer_max_vals(self):
        res = [mpz(self.max_input)]
        for n in self.structure[:-1]:
            layer_max = (((res[-1]*n) + 1)*self.max_weight)**2
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
                x = x**2
        else:
            for mat, vec in zip(self.weights, self.biases):
                x = np.dot(mat, x) + vec
                x = x**2
        return x

    def random_weights(self, norm):
        '''returns random weights between 1 and max_weight'''
        res = []
        n = self.structure[0]
        k = self.max_weight // norm
        w = np.random.randint(
            low = -k-1,
            high = k+1,
            size = (self.structure[1],n)
            )
        res.append(np.array(w, dtype = float))
        for i in range(self.depth - 1):
            w = np.random.randint(
                low = -k-1,
                high = k+1,
                size = (self.structure[i+2],self.structure[i+1])
                )
            res.append(np.array(w, dtype = float))
        return res
        
    def random_biases(self, norm):
        '''returns random biases between 1 and max_weight'''
        res = []
        k = self.max_weight // norm
        b = np.array(
            [[mpz(
                np.random.randint(
                    -k-1,
                    k+1
                    )
                )] for i in range(self.structure[1])]
            )
        res.append(np.array(b, dtype = float))
        for i in range(self.depth - 1):
            b = np.array(
                [[mpz(
                    np.random.randint(
                        -k-1,
                        k+1
                        )
                    )] for i in range(self.structure[i+2])]
                )
            res.append(np.array(b, dtype = float))
        return res
 
    def compute_layers(self, data):
        ''' returns a list of matrices with values in
        each layer after forward propagation
        '''
        resZ = []
        resA = []
        z = 0
        a = 0
        for mat, vec in zip(self.weights, self.biases):
            z = np.dot(mat, a) + vec if resZ else np.dot(mat, data) + vec
            a = z**2
            resZ.append(z)
            resA.append(a)
        return resZ, resA
    
    def compute_deltas(self, layersZ, layersA, lables):
        ''' returns deltas
        '''
        res = []
        for l in reversed(range(len(layersA))):
            if l == len(layersA)-1:
                d = (layersA[l] - lables) * 2*layersZ[l] 
            else:
                d = self.weights[l+1].T.dot(d) * 2*layersZ[l] 
            res.append(d)
        res.reverse()
        return res
    
    
    def compute_gradients(self, layersA, deltas):
        '''returns gradients for weights and biases.
        The first layer in layes is the input.
        '''
        gradW = []
        gradB = []
        for i in range(self.depth):
            # for biases:
            gb = np.sum(deltas[i], axis = 1)/(deltas[i].shape[1])
            gradB.append(gb)
            
            # for weights:
            gw = (layersA[i] @ deltas[i].T)/(deltas[i].shape[1])
            gradW.append(gw)       
        return gradW, gradB
    
    
    def fit(self,X,y,iterations, verbose=1):
        '''X is a database. y is a list of labels. 
        iterations is an integer. 
        '''
        k = self.max_weight
        for i in range(iterations):
            if verbose:
                print(f"Running interation {i+1} / {iterations}.")
            Z, A = self.compute_layers(X)
            d = self.compute_deltas(Z, A, y.T)
            Dw, Db = self.compute_gradients([X] + A, d)
            Dw, Db = fix_gradients(Dw, Db, scaling = self.learning_rate)
            for i in range(self.depth):
                self.weights[i] -=  Dw[i].T
                self.weights[i][self.weights[i]<-k] = -k
                self.weights[i][self.weights[i]>k] = k
                self.biases[i] -=  Db[i].reshape(M.biases[i].shape)
                self.biases[i][self.biases[i]<-k] = -k
                self.biases[i][self.biases[i]>k] = k
        return
    
    def cost(self, data, labels):
        ''' data is a database feedable to self.predict
        lables is a list of wanted outputs as binary vectors.
        returns a float
        '''
        # compute predictions
        Y = self.predict(data)
        norm = labels.shape[0] * labels.shape[1] * 2
        norm = norm ** 2 # pow(2,250)
        c = Y.T - labels
        c = c**2
        c = np.sum(c)
        c = c / norm
        return c / self.beta

# loading mnist data
print("Loading MNIST dataset and arranging data.")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# changing orientation to run predict over entire database at once:
# (to predict a single sample use:  x_train.T[sample_index])
x_train = x_train.reshape(60000,784).transpose()
x_test = x_test.reshape(10000,784).transpose()


# initializing the model 
initializing = 0
batch_training = 1
batch_size = 100
num_batches = 60000//batch_size
epochs = 10

if initializing:
    print("Initializing model.")   
    M = Model(
        max_input = 255,
        max_weight = 100,
        structure = [784,128,10],
        # The initial weights and biases are sampled from 
        # a smaller domain, scaled down according to these
        # factors:
        weights_scaler=1, 
        biases_scaler=1,
        # the learning rate specifies the maximal step taken
        # in every step of the gradient descent algorithm.
        learning_rate = 1
        )
    running_cost = []
    # inspect initial status
    pred = M.predict(x_test).T
    predictions =[np.argmax(pred[i]) for i in range(10000)]
    match = y_test == predictions
    plt.hist(predictions)

# Creating binary vectors for cost computation. 
# Beware! orientation is opposite to predict's output
y_train_bin = np.zeros((60000,10), dtype = mpz)
for val, vec in zip(y_train, y_train_bin):
    vec[val] = M.beta 
y_test_bin = np.zeros((10000,10), dtype = mpz)
for val, vec in zip(y_test, y_test_bin):
    vec[val] = M.beta 


#train model.
if batch_training:
    # split data to minibatches
    x_mini_batches = [
        x_train.T[
            l*batch_size:(l+1)*batch_size].T 
        for l in range(num_batches)
        ]
    y_mini_batches = [
        y_train_bin[
            l*batch_size:(l+1)*batch_size] 
        for l in range(num_batches)
        ]
    
    # fit and plot running cost
    print("Training model.")
    running_cost.append(M.cost(x_test, y_test_bin))
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} / {epochs}...")
        for batch in range(num_batches):                
            M.fit(
                x_mini_batches[batch],
                y_mini_batches[batch],
                1,
                verbose=0
                )
            if batch%20 == 0:
                print(".", sep="", end="")
            
        c = M.cost(x_test,y_test_bin)
        running_cost.append(c)  
        print(f"\nCompleted.\tCost: {c}")
        
    plt.plot(running_cost)

    print("Done training.\nAccuracy:")
    pred = M.predict(x_test).T
    predictions =[np.argmax(pred[i]) for i in range(10000)]
    match = y_test == predictions
    plt.hist(predictions)
    print(f"{sum(match)} out of 10,000.")
    