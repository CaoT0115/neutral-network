# input --> { neuron(fc_cache) --> bn(bn_cache) --> relu(relu_cache) --> dropout_cache --> } softmax loss
#             xi, wi, bi --> hi
#                            hi_cache = fc_cache, bn_cache, relu_cache, dropout_cache
#                                       fc_cache = xi, wi, bi
#                                       bn_cache = x, sample_mean, sample_var, x_hat, eps, gamma, beta 
#                                       relu_cache = a_bn
#                                       dropout_cache = dropout_param, mask
#             self.param{} = wi, bi, gammai, betai, hi
#             self.bn_param{} = mode, eps, momentum, mean, var
#             self.dropout_param{} = mode, p, seed
#             grads{} = w, b, gamma, beta
#             config{} = learning_rate, momentum, velocityi



class network(object):

	def __init__(self,
		         pic_dim = 32*32*3,
		         hidden_dim,
		         class_num = 10,
		         w_scale = 1e-3,
		         reg = 0.0,
		         dropout = 0,
		         seed = None):

	    self.layer_num = len(hidden_dim) + 1
	    self.reg = reg

	    self.param = {}    #forward param

	    in_dim = pic_dim

        for i, out_dim in enumerate(hidden_dim):    #initialize
        	self.param["w%d"%(i+1)] = w_scale * np.random.randn(in_dim, out_dim)
        	self.param["b%d"%(i+1)] = np.zeros((out_dim))

        		self.param["gamma%d"%(i+1)] = np.ones((out_dim))
        		self.param["beta%d"%(i+1)] = np.zeros((out_dim))
        	in_dim = out_dim

       	self.param["w%d"%(self.layer_num)] = w_scale * np.random.randn(in_dim, class_num)
       	self.param["b%d"%(self.layer_num)] = np.zeros(class_num)

       	self.dropout_param = {}

       		self.dropout_param = {"mode": "train", "p": dropout}
       	if seed in not None:
       		self.dropout_param["seed"] = seed

       	self.bn_param = {}

       		self.bn_param = [{"mode": "train"} for i in range(len(hidden_dim))]


    def loss(self, x, y = None):

    	mode = "test" if y is None else "train"

    		self.dropout_param["mode"] = mode
    		for bn_param in self.bn_param:
    			bn_param["mode"] = mode


    	out = x
    	for i in range(self.layer_num-1):
    		out, self.param["h%d_cache"%(i+1)] = neuron_bn_relu_dropout_forward(out, self.param["w%d"%(i+1)], self.param["b%d"%(i+1)], self.param["gamma%d"%(i+1)], self.param["beta%d"%(i+1)], self.bn_param[i], self.dropout_param[i])

    	scores, scores_cache = neuron_forward(out, self.param["w%d"%(self.layer_num)], self.param["b%d"%(self.layer_num)])

    	if mode == "test":
    		return scores   #stop when test


    	loss, grads = 0, {}

    	loss, dscores = softmax_loss_back(scores, y)

    	loss += 0.5 * self.reg * np.sum(self.param["w%d"%(self.layer_num)]**2)    #normalize loss

    	dout, grads["w%d" % (self.layer_num)], grads["b%d" % (self.layer_num)] = neuron_back(dscores, scores_cache)


    	for i in range(self.layer_num-1):
    		j = self.layer_num-1 - i
    		loss += 0.5 * self.reg * np.sum(self.param["w%d"%(j)]**2)    #normalize loss
    		dout, grads["w%d" % (j)], grads["b%d" % (j)], grads["gamma%d"%(j)], grads["beta%d"%(j)] = neuron_bn_relu_dropout_back(dout, self.param["h%d_cache"%(j)])
    		grads["w%d" % (j)] += self.reg * grads["w%d" % (j)]

        return loss, grads


    def descent(grads, config=None):
    	if config is None:
		    config = {}
		    config.setdefault("learning_rate", 1e‐2)
		    config.setdefault("momentum", 0.9)

	    self.param["w%d"%(self.layer_num)] -= config["learning_rate"] * grads["w%d"%(self.layer_num)]
    	self.param["b%d"%(self.layer_num)] -= config["learning_rate"] * grads["b%d"%(self.layer_num)]

    	for i in range(self.layer_num-1):
    		w = self.param["w%d"%(i+1)]
    		b = self.param["b%d"%(i+1)]
    		gamma = self.param["gamma%d"%(i+1)]
    		beta = self.param["beta%d"%(i+1)]
    		dw = grads["w%d"%(i+1)]
    		db = grads["b%d"%(i+1)]
    		dgamma = grads["gamma%d"%(i+1)]
    		dbeta = grads["beta%d"%(i+1)]

    		self.param["w%d"%(i+1)], self.param["b%d"%(i+1)], self.param["gamma%d"%(i+1)], self.param["beta%d"%(i+1)], config = sgd(w, b, gamma, beta, dw, db, dgamma, dbeta, config)

    		v = config.get("velocity%d"%(i+1), np.zeros_like(w))
    		v = config["momentum"] * v ‐ config["learning_rate"] * dw
    		w += v

    		config["velocity%d"%(i+1)] = v

    	return config





def neuron_forward(x, w, b):

    out = None       
    reshaped_x = np.reshape(x, (x.shape[0],‐1))         
    out = reshape_x.dot(w) +b                            
    cache = (x, w, b)   
                                  
    return out, cache


def batchnorm_forward(x, gamma, beta, bn_param):
	mode = bn_param["mode"]
	eps = bn_param.get("eps", 1e-5)
	momentum = bn_param.get("momentum", 0.9)

	N, D = x.shape
	mean = bn_param.get("mean", np.zeros(D, dtype = x.dtype))
	var = bn_param.get("var", np.zeros(D, dtype = x.dtype))

	out, cache = None, None

	if mode == "train":
		sample_mean = np.mean(x, axis=0)
		sample_var = np.var(x, axis=0)
		x_hat = (x ‐ sample_mean) / (np.sqrt(sample_var + eps))

		out = gamma * x_hat + beta
		cache = (x, sample_mean, sample_var, x_hat, eps, gamma, beta)

		mean = momentum * mean + (1 ‐ momentum) * sample_mean
		running_var = momentum * var + (1 ‐ momentum) * sample_var

	elif mode == "test":
		out = (x ‐ running_mean) * gamma / (np.sqrt(running_var + eps)) + beta

	else:
		print("invalid bn mode")
		exit()

	bn_param["mean"], bn_param["var"] = mean, var

	return out, cache


def relu_forward(x):

	out = np.maximum(0, x)   
	cache = x     
	    
    return out, cache


def dropout_forward(x, dropout_param):
	p, mode = dropout_param["p"], dropout_param["mode"]
	if "seed" in dropout_param:
		np.random.seed(dropout_param["seed"])
	mask, out = None, None

	if mode == "train":
		active_prob = 1 - p
		mask = (np.random.rand(x.shape) < active_prob) / active_prob
		out = mask * x

	elif mode == "test":
		out = x

	cache = (dropout_param, mask)
	out = out.astype(x.dtype, copy=False)

	return out, cache


def neuron_bn_relu_dropout_forward(x, w, b, gamma, beta, bn_param, dropout_param):

	a, fc_cache = neuron_forward(x, w, b)
	a_bn, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
	a_relu, relu_cache = relu_forward(a_bn)
	out, dropout_cache = dropout_forward(a_relu, dropout_param)

	cache = (fc_cache, bn_cache, relu_cache, dropout_cache)

	return out, cache


def softmax_loss_back(x, index):

    probs = np.exp(x)
	probs /= np.sum(probs, axis=1, keepdims=True)

	n = x.shape[0]
	loss = -np.sum(np.log(probs[np.arrange(n), index]))/n

	dz = probs.copy()
	dz[np.arange(n),y] -= 1
	dz /= n

	return loss, dz


def dropout_backward(dout, dropout_cache):
	dropout_param, mask = dropout_cache
	mode = dropout_param["mode"]

	dx = None

	if mode == "train":
		dx = mask * dout

	elif mode == "test":
		dx = dout

	return dx


def relu_back(dout, relu_cache):

	dx, x = None, relu_cache
	dx = (x > 0) * dout

	return dx


def batchnorm_backward(dout, bn_cache):
	x, mean, var, x_hat, eps, gamma, beta = bn_cache
	N = x.shape[0]
	dgamma = np.sum(dout * x_hat, axis=0)  
	dbeta = np.sum(dout * 1.0, axis=0)     
	dx_hat = dout * gamma                  
	dx_hat_numerator = dx_hat / np.sqrt(var + eps)    
	dx_hat_denominator = np.sum(dx_hat * (x ‐ mean), axis=0)    
	dx_1 = dx_hat_numerator            
	dvar = ‐0.5 * ((var + eps) ** (‐1.5)) * dx_hat_denominator 
	dmean = ‐1.0 * np.sum(dx_hat_numerator, axis=0) + \
	dvar * np.mean(‐2.0 * (x ‐ mean), axis=0)  
	dx_var = dvar * 2.0 / N * (x ‐ mean)  
	dx_mean = dmean * 1.0 / N             
	dx = dx_1 + dx_var + dx_mean 

	return dx, dgamma, dbeta


def neuron_back(out, fc_cache):
	
	z, w, b = fc_cache	
	dz, dw, db = None, None, None
	
	reshaped_x = np.reshape(z, (z.shape[0], ‐1))
	
	dz = np.reshape(dout.dot(w.T), z.shape)    
	dw = (reshaped_x.T).dot(dout)
	db = np.sum(dout, axis=0)

	dw = self.reg * dw    #normalize

	return dz, dw, db


def neuron_bn_relu_dropout_back(dout, cache):
	
	fc_cache, bn_cache, relu_cache, dropout_cache = cache
	da_relu = dropout_backward(dout, dropout_cache)
    da_bn = relu_back(da_relu, relu_cache)
    da, dgamma, dbeta = batchnorm_backward(da_bn, bn_cache)
    dx, dw, db = neuron_back(da, fc_cache)
    
    return dx, dw, db, dgamma, dbeta


def sgd(w, b, gamma, beta, dw, db, dgamma, dbeta, config):

	w -= config["learning_rate"] * dw
    b -= config["learning_rate"] * db
	gamma -= config["learning_rate"] * dgamma
	beta -= config["learning_rate"] * dbeta

	return w, b, gamma, beta, config


