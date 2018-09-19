class TwoLayer(object): 
	def __init__(self,
	            input_dim = 28*28, # data dimension   
	            hidden_dim = 100, # nuerons in hidden layer      
	            num_classes = 10, # labels        
	            weight_scale = 1e‐3,
	            reg = 0.0): # weight initial scale

        self.params = {}    # save w, b

        self.params["W1"] = weight_scale * np.random.randn(input_dim,hidden_dim)
        self.params["b1"] = np.zeros((hidden_dim,))
        self.params["W2"] = weight_scale * np.random.randn(hidden_dim,num_classes)
        self.params["b2"] = np.zeros((num_classes,))


    def loss(self, X, y):

    	loss, grads = 0, {},   # save gradient, sgd parameters

    	h1_out, h1_cache = affine_relu_forward(X, self.params["W1"], self.params["b1"])
    	scores, out_cache = affine_forward(h1_out, self.params["W2"], self.params["b2"])

    	loss, dout = softmax_loss(scores, y)


    	loss += 0.5 * self.reg * (np.sum(self.params["W1"] ** 2) + np.sum(self.params["W2"] ** 2))    # regular

    	dW2 += self.reg * self.params["W2"]
    	dW1 += self.reg * self.params["W1"]


    	dout, dw2, db2 = affine_backward(dout, out_cache)
    	grads["W2"] = dw2 , grads["b2"] = db2
    	_, dw1, db1 = affine_relu_backward(dout, h1_cache)
    	grads["W1"] = dw1 , grads["b1"] = db1

    	return loss, grads



def affine_forward(x, w, b):
	"""
	Inputs: x: shape (N, D)  w: shape (D, M)  b: shape (M) 
	Returns a tuple of: out: output, of shape (N, M) cache: (x, w, b)
	"""

    out = None  
    reshaped_x = np.reshape(x, (x.shape[0],‐1))      # N samples, each sample D elements
    out = reshape_x.dot(w) + b       # out = w x +b                            
    fc_cache = (x, w, b) 
                                   
    return out, fc_cache



def relu_forward(x):
    """
    Input: x: Inputs, of any shape
    Returns a tuple of: out, cache: x
    """
    out = np.maximum(0, x)      # remove negative
    relu_cache = x         
    
    return out, relu_cache



def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU
    Inputs: x, w, b
    Returns a tuple of: out, cache
    """
    a, fc_cache = affine_forward(x, w, b) 
    out, relu_cache = relu_forward(a)  
    cache = (fc_cache, relu_cache)  
    
    return out, cache



def softmax_loss(z, y):
    """
    Computes the loss and gradient for softmax classification.
    Inputs: z: shape (N, C), z[i, j]: the score for the jth class for the ith input, y: shape (N) where y[i] is the label for x[i] and
    Returns a tuple of: loss,  dz: Gradient of the loss
    """
    probs = np.exp(z ‐ np.max(z, axis=1, keepdims=True))     # each sample minus highest score
    probs /= np.sum(probs, axis=1, keepdims=True)          # prob for each label
    N = z.shape[0]                                          # number of first dimmension, number of samples
    loss = ‐np.sum(np.log(probs[np.arange(N), y])) / N     # average scores of right labels
    
    dz = probs.copy()
    dz[np.arange(N), y] ‐= 1         # right labels minus 1
    dz /= N
    
    return loss, dz



def affine_backward(dout, fc_cache):
    """
    Computes the backward pass for an affine layer.
    Inputs: dout: shape (N, M), cache: x, w, b

    Returns a tuple of: dx, dw, db
    """
    x, w, b = fc_cache
    dx, dw, db = None, None, None
    reshaped_x = np.reshape(x, (x.shape[0], ‐1))
    
    dx = np.reshape(dout.dot(w.T), x.shape)    
    dw = (reshaped_x.T).dot(dout)
    db = np.sum(dout, axis=0)
    
    return dx, dw, db



def relu_backward(dout, relu_cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).
    Input: dout, cache
    Returns: dx
    """
    dx, x = None, relu_cache
    dx = (x > 0) * dout      # forward relu with element > 0, gradient is 1, else 0
    
    return dx



def affine_relu_backward(dout, cache):
	"""
	Backward pass for the affine‐relu convenience layer
	"""
    fc_cache, relu_cache = cache            # fc_cache = (x, w, b)   relu_cache = a
    da = relu_backward(dout, relu_cache)    # da = (x > 0) * relu_cache
    dx, dw, db = affine_backward(da, fc_cache)
    
    return dx, dw, db



def sgd_momentum(w, dw, config = None):
	if config is None: config = {}
	    config.setdefault('learning_rate', 1e‐2)
	    config.setdefault('momentum', 0.9)
	
	v = config.get('velocity', np.zeros_like(w))
	next_w = None
	
	v = config["momentum"] * v ‐ config["learning_rate"] * dw
	next_w = w + v
	config['velocity'] = v

	return next_w, config