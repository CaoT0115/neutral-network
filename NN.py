# input --> { neuron(fc_cache) --> relu(relu_cache) --> } softmax loss
#             wi, bi --> hi

class network(object):

	def __init__(self,
		         pic_dim = 32*32*3,
		         hidden_dim,
		         class_num = 10,
		         w_scale = 1e-3):

	    self.layer_num = len(hidden_dim) + 1
	    self.param = {}

	    in_dim = pic_dim

        for i, out_dim in enumerate(hidden_dim):
        	self.param["w%d"%(i+1)] = w_scale * np.random.randn(in_dim, out_dim)
        	self.param["b%d"%(i+1)] = np.zeros(out_dim)
        	in_dim = out_dim

       	self.param["w%d"%(self.layer_num)] = w_scale * np.random.randn(in_dim, class_num)
       	self.param["b%d"%(self.layer_num)] = np.zeros(class_num)


    def loss(self, x, y):

    	loss, grads = 0, {}

    	out = x
    	for i in range(self.layer_num-1):
    		out, self.param["h%d_cache"%(i+1)] = neuron_relu_forward(out, self.param["w%d"%(i+1)], self.param["b%d"%(i+1)])

    	scores, scores_cache = neuron_forward(out, self.param["w%d"%(self.layer_num)], self.param["b%d"%(self.layer_num)])

    	loss, dscores = softmax_loss_back(scores, y)

    	dout, grads["W%d" % (self.num_layers)], grads["b%d" % (self.num_layers)] = neuron_back(dscores, scores_cache)


    	for i in range(self.layer_num-1):
    		j = self.layer_num-1 - i
    		dout, grads["W%d" % (j)], grads["b%d" % (j)] = neuron_back(dout, self.param["h%d_cache"%(j)])

        return loss, grads




def neuron_forward(x, w, b):

    out = None       
    reshaped_x = np.reshape(x, (x.shape[0],‐1))         
    out = reshape_x.dot(w) +b                            
    cache = (x, w, b)   
                                  
    return out, cache


def relu_forward(x):

	out = np.maximum(0, x)   
	cache = x     
	    
    return out, cache


def neuron_relu_forward(x, w, b):

	a, fc_cache = neuron_forward(x, w, b)
	out, relu_cache = relu_forward(a)
	cache = (fc_cache, relu_cache)

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


def neuron_back(out, fc_cache):
	
	z, w, b = fc_cache	
	dz, dw, db = None, None, None
	
	reshaped_x = np.reshape(z, (z.shape[0], ‐1))
	
	dz = np.reshape(dout.dot(w.T), z.shape)    
	dw = (reshaped_x.T).dot(dout)
	db = np.sum(dout, axis=0)

	return dz, dw, db



def relu_back(dout, relu_cache):

	dx, x = None, relu_cache
	dx = (x > 0) * dout

	return dx


def neuron_relu_back(dout, cache):
	
	fc_cache, relu_cache = cache
    dr = relu_back(dout, relu_cache)
    dx, dw, db = neuron_back(dr, fc_cache)

    return dx, dw, db