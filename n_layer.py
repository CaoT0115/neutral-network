from mlxtend.data import loadlocal_mnist
import numpy as np

X, y = loadlocal_mnist(
        images_path="D:\\ctc\\ureca\\mnist\\train-images.idx3-ubyte", 
        labels_path="D:\\ctc\\ureca\\mnist\\train-labels.idx1-ubyte")

class TwoLayer(object):
    def __init__(self,
       input_dim = 28*28, # data dimension   
       hidden_dim = 100, # nuerons in hidden layer      
       num_classes = 10, # labels 
       num_hlayer = 10,
       weight_scale = 7e-3,
       reg = 0.1): # weight initial scale
        
        self.reg = reg
        self.params = {}   # save w, b
        self.config = {}
        self.num_hlayer = num_hlayer
        
        in_dim = input_dim
        for i in range(num_hlayer):
            self.params["W%d"%(i+1)] = weight_scale * np.random.randn(in_dim,hidden_dim)
            self.params["b%d"%(i+1)] = np.zeros((hidden_dim,))
            in_dim = hidden_dim
        
        self.params["W%d"%(num_hlayer+1)] = weight_scale * np.random.randn(hidden_dim,num_classes)
        self.params["b%d"%(num_hlayer+1)] = np.zeros((num_classes,))



    def loss(self, X, y):

        loss, grads,f_cache = 0, {}, {},   # save gradient, sgd parameters
        
        out = X
        for i in range(self.num_hlayer):
            j = i+1
            w, b = self.params["W%d"%(j)], self.params["b%d"%(j)]
            out, f_cache["h%d"%(j)] = affine_relu_forward(out, w, b)
        
        j = j+1
        w = self.params["W%d"%(self.num_hlayer+1)]
        b = self.params["b%d"%(self.num_hlayer+1)]
        scores, out_cache = affine_forward(out, w, b)

        loss, dout = softmax_loss(scores, y)

        loss += 0.5 * self.reg * (np.sum(self.params["W%d"%(self.num_hlayer+1)] ** 2))# regular


        dout, dw, db = affine_backward(dout, out_cache)
        grads["W%d"%(self.num_hlayer+1)] = dw + self.reg * self.params["W%d"%(self.num_hlayer+1)]
        grads["b%d"%(self.num_hlayer+1)] = db
        
        for i in range(self.num_hlayer):
            j = self.num_hlayer - i
            loss += 0.5 * self.reg * (np.sum(self.params["W%d"%(j)] ** 2))
        dout, dw, db = affine_relu_backward(dout, f_cache["h%d"%(j)])
        grads["W%d"%(j)] = dw + self.reg * self.params["W1"] 
        grads["b%d"%(j)] = db

        return loss, grads



def affine_forward(x, w, b):
    """
    Inputs: x: shape (N, D)  w: shape (D, M)  b: shape (M) 
    Returns a tuple of: out: output, of shape (N, M) cache: (x, w, b)
    """

    out = None 
    reshaped_x = np.reshape(x, (x.shape[0],-1))  # N samples, each sample D elements
    out = reshaped_x.dot(w) + b   # out = w x +b                            
    fc_cache = (x, w, b) 


    return out, fc_cache



def relu_forward(x):
    """
    Input: x: Inputs, of any shape
    Returns a tuple of: out, cache: x
    """
    out = np.maximum(0, x)      # remove negative
    relu_cache = x

    return out, relu_cache



def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU
    Inputs: x, w, b
    Returns a tuple of: out, cache
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)

    return out, cache



def softmax_loss(z, y):
    """
    Computes the loss and gradient for softmax classification.
    Inputs: z: shape (N, C), z[i, j]: the score for the jth class for the ith input, y: shape (N) where y[i] is the label for x[i] and
    Returns a tuple of: loss,  dz: Gradient of the loss
    """
    probs = np.exp(z - np.max(z, axis=1, keepdims=True)) # each sample minus highest score
    probs /= np.sum(probs, axis=1, keepdims=True)        # prob for each label
    N = z.shape[0]                  # number of first dimmension, number of samples
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N        # average scores of right labels

    dz = probs.copy()
    dz[np.arange(N), y] -= 1 # right labels minus 1
    dz /= N

    return loss, dz



def affine_backward(dout, fc_cache):
    """
    Computes the backward pass for an affine layer.
    Inputs: dout: shape (N, M), cache: x, w, b

    Returns a tuple of: dx, dw, db
    """
    x, w, b = fc_cache
    dx, dw, db = None, None, None
    reshaped_x = np.reshape(x, (x.shape[0], -1))

    dx = np.reshape(dout.dot(w.T), x.shape)
    dw = (reshaped_x.T).dot(dout)
    db = np.sum(dout, axis=0)

    return dx, dw, db



def relu_backward(dout, relu_cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).
    Input: dout, cache
    Returns: dx
    """
    dx, x = None, relu_cache
    dx = (x > 0) * dout        # forward relu with element > 0, gradient is 1, else 0

    return dx



def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine‐relu convenience layer
    """
    fc_cache, relu_cache = cache           # fc_cache = (x, w, b)   relu_cache = a
    da = relu_backward(dout, relu_cache)     # da = (x > 0) * relu_cache
    dx, dw, db = affine_backward(da, fc_cache)

    return dx, dw, db



def sgd_momentum(w, dw, b, db, config = None):
    if config is None: config = {}
    config.setdefault("learning_rate", 0.003)
    config.setdefault("momentum", 0.86)
    
    v = config.get("velocity_w", np.zeros_like(w))
    next_w = None
    u = config.get("velocity_b", np.zeros_like(b))
    next_b = None
    
    v = config["momentum"] * v - config["learning_rate"] * dw
    next_w = w + v
    u = config["momentum"] * u - config["learning_rate"] * db
    next_b = b + u
    
    config["velocity_w"] = v
    config["velocity_b"] = u

    return next_w, next_b, config


nn = TwoLayer()

loss, grads = nn.loss(X,y)
for i in range(nn.num_hlayer+1):
    nn.params["W%d"%(i+1)], nn.params["b%d"%(i+1)], nn.config["config%d"%(i+1)] = sgd_momentum(nn.params["W%d"%(i+1)], grads["W%d"%(i+1)], nn.params["b%d"%(i+1)], grads["b%d"%(i+1)])
#print(loss)

for k in range(10):
#    for j in range(10):
        loss, grads = nn.loss(X,y)
        for i in range(nn.num_hlayer+1):
            nn.params["W%d"%(i+1)], nn.params["b%d"%(i+1)], nn.config["config%d"%(i+1)] = sgd_momentum(nn.params["W%d"%(i+1)], grads["W%d"%(i+1)], nn.params["b%d"%(i+1)], grads["b%d"%(i+1)], nn.config["config%d"%(i+1)])
        print(k)
        print(loss)




        

        Xt, yt = loadlocal_mnist(
        images_path="D:\\ctc\\ureca\\mnist\\t10k-images.idx3-ubyte", 
        labels_path="D:\\ctc\\ureca\\mnist\\t10k-labels.idx1-ubyte")

        pred = []
        h1_out, h1_cache = affine_relu_forward(Xt, nn.params["W1"], nn.params["b1"])
        scores, out_cache = affine_forward(h1_out, nn.params["W2"], nn.params["b2"])
        pred = np.argmax(scores, axis=1)

        acc = np.mean(pred == yt)
        print(acc)
