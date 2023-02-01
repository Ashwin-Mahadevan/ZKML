import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from datetime import datetime


def activation(inp):
    return 1.0/(1.0 + np.exp(-inp))

def act_prime(inp):
    return (1-activation(inp))*activation(inp)


def fix_gradients(DW, DB, scaling = 0.25):
    W = [w.copy() for w in DW]
    B = [b.copy() for b in DB]
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
    return res

class Model:
    def __init__(
            self,
            structure = [784, 16, 16, 10],
            learning_rate = 0.1
            ):
        """
        max_input is an integer.
        max weight is some small int
        structure is a list that indicates the lengths 
        of layers from the input to output (discluding biases).
        Defines a model
        """
        self.structure= np.array(structure)
        self.depth = len(self.structure)-1
        self.weights = [np.random.rand(
            self.structure[i+1],
            self.structure[i]
            )*2 - 1 for i in range(self.depth)]
        self.biases = [np.random.rand(n,1)*2 - 1
            for n in self.structure[1:]]
        self.learning_rate = learning_rate
        
    @classmethod
    def from_weights_and_biases(cls, WB_pair, lrn):
        '''WB_pair is a pair
        the first entry is the list of weights
        the second entry is list of biases
        retuens a Model object.'''
        weights = WB_pair[0]
        biases  = WB_pair[1]
        strc = [w.shape[1] for w in weights]
        strc.append(weights[-1].shape[0])
        model = cls(strc, lrn)
        model.weights = weights
        model.biases  = biases
        return model
        
    
    def predict(self, inp):
        x=inp.copy()
        if len(x.shape)==1:
            for i in range(self.depth):
                x = (
                    self.weights[i].dot(x).reshape(-1) + 
                    self.biases[i].reshape(-1)
                    )
                x = activation(x)
        else:
            for mat, vec in zip(self.weights, self.biases):
                x = np.dot(mat, x) + vec
                x = activation(x)
        return x

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
            a = activation(z)
            resZ.append(z)
            resA.append(a)
        return resZ, resA
    
    def compute_deltas(self, layersZ, layersA, lables):
        ''' returns deltas
        '''
        res = []
        for l in reversed(range(len(layersA))):
            if l == len(layersA)-1:
                d = (layersA[l] - lables) * act_prime(layersZ[l])
            else:
                d = act_prime(layersZ[l]) * self.weights[l+1].T.dot(d)
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
        for i in range(iterations):
            if verbose:
                print(f"Running interation {i+1} / {iterations}.")
            Z, A = self.compute_layers(X)
            d = self.compute_deltas(Z, A, y.T)
            Dw, Db = self.compute_gradients([X] + A, d)
            # Dw, Db = fix_gradients(Dw, Db, scaling = self.learning_rate)
            for i in range(self.depth):
                self.weights[i] -=  Dw[i].T
                self.biases[i]  -=  Db[i].reshape(M.biases[i].shape)
        return
    
    def cost(self, data, labels):
        ''' data is a database feedable to self.predict
        lables is a list of wanted outputs as binary vectors.
        returns a float
        '''
        # compute predictions
        Y = self.predict(data)
        norm = labels.shape[0] * labels.shape[1] * 2
        c = Y.T - labels
        c = c**2
        c = np.sum(c)
        c = c / norm
        return c 

# =============================================================================
# # =============================================================================
# # ###  S C R I P T   B E G I N S   H E R E 
# # =============================================================================
# =============================================================================

#=============================================================================
# # what do we do now.
#=============================================================================
load_and_arrange_data = 1
initialize_new_model = 1
load_model_from_memory = 0
batch_training = 1
improve_cost = 0
improve_accuracy = 0
save_model = 0

#=============================================================================
# # # #  Parameters:
#=============================================================================
# data parameters:
label_mark = 1# float(pow(2,50))
train_samples = 50000
sample_size = 32*32*3
input_range = 255
test_samples = 10000
outputs = 10
# model parameters:
struct = [sample_size, 320, 160, 80, 40, 20, 10]
lr = 0.25
loc = 'my_model_2023-01-30_21-41-21.pickle'
# training parameters:
batch_size = 1000
num_batches = train_samples//batch_size
iterations = 1
epochs = 50


# =============================================================================
# # load_and_arrange_data
# =============================================================================
if load_and_arrange_data:
    print("Loading MNIST dataset and arranging data.")
    mnist = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # changing orientation to run predict over entire database at once:
    # (to predict a single sample use:  x_train.T[sample_index])
    x_train = x_train.reshape(train_samples,sample_size).transpose() / input_range
    x_test = x_test.reshape(test_samples,sample_size).transpose() / input_range
    y_train_bin = np.zeros((train_samples,outputs), dtype = float)
    for val, vec in zip(y_train, y_train_bin):
        vec[val] = label_mark
    y_test_bin = np.zeros((test_samples,outputs), dtype = float)
    for val, vec in zip(y_test, y_test_bin):
        vec[val] = label_mark


# =============================================================================
# # initialzie a new model
# =============================================================================
if initialize_new_model:
    print("Initializing model.")   
    M = Model(
        structure = struct,
        learning_rate = lr
        )
    running_cost = []
    # inspect initial status
    pred = M.predict(x_test).T
    predictions =[np.argmax(pred[i]) for i in range(test_samples)]
    match = y_test == predictions
    
    
# =============================================================================
# # load model from memory
# =============================================================================
if load_model_from_memory:
    print(f"Loading model from memory: {loc}")
    with open(loc, 'rb') as file:
        W_and_B = pickle.load(file)
    M = Model.from_weights_and_biases(W_and_B, lr)
    running_cost = []
    # inspect initial status
    pred = M.predict(x_test).T
    predictions =[np.argmax(pred[i]) for i in range(test_samples)]
    match = y_test == predictions
    accuracy = sum(match)
    print(f"Loaded model from memory.\n accuracy: {sum(match)/10000}.")
        

# =============================================================================
# #train model.
# =============================================================================
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
                iterations,
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
    predictions =np.array([np.argmax(pred[i]) for i in range(test_samples)]).reshape(y_test.shape)
    match = y_test == predictions
    accuracy = sum(match)
    print(f"{accuracy} out of 10,000.")



# =============================================================================
# #improve cost
# =============================================================================
if improve_cost:
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
    print("Improving cost.")
    min_cost = min(running_cost)
    keep_going = True
    count = 1
    while keep_going:
        print(f"Epoch #{count}.")
        count +=1
        for batch in range(num_batches):                
            M.fit(
                x_mini_batches[batch],
                y_mini_batches[batch],
                iterations,
                verbose=0
                )
            if batch%20 == 0:
                print(".", sep="", end="")
            
        c = M.cost(x_test,y_test_bin)
        running_cost.append(c)  
        if c < min_cost:
            keep_going = False
            print(f"\nGot better cost: {c}")
        
    plt.plot(running_cost)

    print("Accuracy:")
    pred = M.predict(x_test).T
    predictions =[np.argmax(pred[i]) for i in range(test_samples)]
    match = y_test == predictions
    accuracy = sum(match)
    print(f"{accuracy} out of 10,000.")


# =============================================================================
# #improve accuracy
# =============================================================================
if improve_accuracy:
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
    print("Improving accuracy.")
    best = accuracy
    keep_going = True
    count = 1
    while keep_going:
        print(f"Epoch #{count}.")
        count +=1
        for batch in range(num_batches):                
            M.fit(
                x_mini_batches[batch],
                y_mini_batches[batch],
                iterations,
                verbose=0
                )
            if batch%20 == 0:
                print(".", sep="", end="")
            
        pred = M.predict(x_test).T
        predictions =[np.argmax(pred[i]) for i in range(test_samples)]
        match = y_test == predictions  
        accuracy = sum(match)
        c = M.cost(x_test,y_test_bin)
        running_cost.append(c)  
        if accuracy > best:
            keep_going = False
            best = accuracy
            print(f"\nGot better accuracy: {best}")
            plt.plot(running_cost)
        

# =============================================================================
# # save model to memory
# =============================================================================
if save_model:
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f'my_model_{time_str}.pickle', 'wb') as file:
        pickle.dump((M.weights, M.biases), file)
    