import numpy as np
import tensorflow as tf
from scipy.special import gamma, factorial
import matplotlib.pyplot as plt
import scipy.optimize
from time import time
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

DTYPE='float32'
tmin = 0.01
tmax = 5.
N_d = 100
param_exact = [0.9, 0.1, 1., 20.] #alpha, beta, E, tau
alpha, beta, E, tau = np.float32(np.array(param_exact))
lb = tf.constant([tmin], dtype=DTYPE)
ub = tf.constant([tmax], dtype=DTYPE)

t_d = tf.reshape(tf.linspace(lb[0], ub[0], N_d),(-1,1))

def Mittag_Leffler(t_d, kappa, mu):
    E = 0.0
    for n in range(0,100):
        E += t_d**n/gamma(kappa*n + mu)
    return E
kappa = alpha - beta
mu = 1. - beta
G = E*(t_d/tau)**(-beta)*Mittag_Leffler(-(t_d/tau)**kappa, kappa, mu)
X_data = t_d
u_data = G

del alpha

#FORWARD
# Set data type
tf.keras.backend.set_floatx(DTYPE)
tf.random.set_seed(42)
log10 = tf.experimental.numpy.log10

N_r = 100
Shuffle = False

t_r = tf.reshape(tf.linspace(lb[0], ub[0], N_r),(-1,1))
h = [t_r[i] - t_r[i-1] for i in range(1, len(t_r))]
h = np.insert(h, 0, (t_r[1] - t_r[0]))
h = h.astype('float32')
h = h[0]  

X_r = tf.concat([t_r], axis=1)

if Shuffle:
    X_r = tf.random.shuffle(X_r)
    
# Define model architecture
class PINN_NeuralNet(tf.keras.Model):
    """ Set basic architecture of the PINN model."""

    def __init__(self, lb, ub, 
            output_dim=1,
            num_hidden_layers=4, 
            num_neurons_per_layer=10,
            activation='tanh',
            kernel_initializer='glorot_normal',
            **kwargs):
        super().__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim
        self.lambd = tf.Variable(self.CM(CMC)[0]*tf.ones(self.CM(CMC)[1]), trainable=True, dtype=DTYPE,
                                 constraint=lambda x: tf.clip_by_value(x, self.CM(CMC)[2], self.CM(CMC)[3]))
        
        self.lambd_list = []
        self.lb = lb
        self.ub = ub
        
        # Define NN architecture
        self.scale = tf.keras.layers.Lambda(
            lambda x: 2.0*(x - lb)/(ub - lb) - 1.0)
        self.hidden = [tf.keras.layers.Dense(num_neurons_per_layer,
                             activation=tf.keras.activations.get(activation),
                             kernel_initializer=kernel_initializer)
                           for _ in range(self.num_hidden_layers)]
        self.out = tf.keras.layers.Dense(output_dim)
        
    def CM(self, CMC):          #Constitutive model
        if CMC == 1: 
            num_param = 2       #Number of parameters to be recovered
            init = [.25, .15]   #Initial values
            low = [1e-2, 1e-2]  #Lowest possible value
            high = [.99, .99]   #Highest possible value
            return init, num_param, low, high
    
    def call(self, X):
        """Forward-pass through neural network."""
        Z = X
        for i in range(self.num_hidden_layers):
            Z = self.hidden[i](Z)
        return self.out(Z)

#Define solver
class PINNSolver():
    def __init__(self, model, X_r):
        self.model = model
        # Store collocation points
        self.t = X_r
        
        # Initialize history of losses and global iteration counter
        self.hist = []
        self.iter = 0
        self.fractional = []


    def Caputo(self, alpha, f, data_point):   #Caputo method for fractional derivative
        summation = 0.
        sigma = 0.
        for k in range (data_point+1):
            if k==0:
                sigma = 1.
            elif 0<k<data_point:
                sigma = (k-1.)**(1-alpha) - 2.*k**(1.-alpha) + (k+1.)**(1.-alpha)                
            elif k==data_point:
                sigma = (k-1.)**(1.-alpha)-k**(1.-alpha)+(1. - alpha)*k**-alpha
            summation += sigma*(f[data_point-k])
        fractional = h**(-alpha) * 1./tf.exp(tf.math.lgamma(2. - alpha))*summation
        return fractional
    
    def fun_r(self, t, u):     #Computes residual of constitutive equation
        alpha, beta = [self.model.lambd[j] for j in range(self.model.CM(CMC)[1])]
        tf.reshape(alpha, (-1,1))
        tf.reshape(beta, (-1,1))
        frac1 = [self.Caputo(alpha - beta, u, index) for index in range (N_r)]
        res = tf.pow(tau, alpha - beta)*frac1 +\
    u - E*tf.pow(tau, alpha)*tf.pow(t, -alpha)/tf.exp(tf.math.lgamma(1. - alpha))
        
        return res
    
    
    def get_r(self):           #Gets residual 
        u = self.model(self.t)      
        return self.fun_r(self.t, u)
    
    def loss_fn(self, X, u):       
        # Compute phi_r
        r = self.get_r()
        Loss_eq = tf.reduce_mean(tf.square(r))
        y_pred = self.model(X)
        Loss_data = tf.reduce_mean(tf.square(u - y_pred))
    
        return 10.*Loss_eq + Loss_data 
    
    def get_grad(self, X, u):
        with tf.GradientTape(persistent=True) as tape:
            # This tape is for derivatives with
            # respect to trainable variables
            tape.watch(self.model.trainable_variables)
            loss = self.loss_fn(X, u)            
        g = tape.gradient(loss, self.model.trainable_variables)
        del tape       
        return loss, g
    
    def solve_with_TFoptimizer(self, optimizer, X, u, N=1001):
        """This method performs a gradient descent type optimization."""      
        @tf.function
        def train_step():
            loss, grad_theta = self.get_grad(X, u)           
            # Perform gradient descent step
            optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            return loss        
        loss = 1.
        while loss>1e-5:
            loss = train_step()            
            self.current_loss = loss.numpy()
            self.callback()

    def solve_with_ScipyOptimizer(self, X, u, method='L-BFGS-B', **kwargs):
        """This method provides an interface to solve the learning problem
        using a routine from scipy.optimize.minimize.
        (Tensorflow 1.xx had an interface implemented, which is not longer
        supported in Tensorflow 2.xx.)
        Type conversion is necessary since scipy-routines are written in Fortran
        which requires 64-bit floats instead of 32-bit floats."""
        
        def get_weight_tensor():
            """Function to return current variables of the model
            as 1d tensor as well as corresponding shapes as lists."""
            
            weight_list = []
            shape_list = []
            
            # Loop over all variables, i.e. weight matrices, bias vectors and unknown parameters
            for v in self.model.variables:
                shape_list.append(v.shape)
                weight_list.extend(v.numpy().flatten())
                
            weight_list = tf.convert_to_tensor(weight_list)
            return weight_list, shape_list

        x0, shape_list = get_weight_tensor()
        
        def set_weight_tensor(weight_list):
            """Function which sets list of weights
            to variables in the model."""
            idx = 0
            for v in self.model.variables:
                vs = v.shape
                
                # Weight matrices
                if len(vs) == 2:  
                    sw = vs[0]*vs[1]
                    new_val = tf.reshape(weight_list[idx:idx+sw],(vs[0],vs[1]))
                    idx += sw
                
                # Bias vectors
                elif len(vs) == 1:
                    new_val = weight_list[idx:idx+vs[0]]
                    idx += vs[0]
                    
                # Variables (in case of parameter identification setting)
                elif len(vs) == 0:
                    new_val = weight_list[idx]
                    idx += 1
                    
                # Assign variables (Casting necessary since scipy requires float64 type)
                v.assign(tf.cast(new_val, DTYPE))
        
        def get_loss_and_grad(w):
            """Function that provides current loss and gradient
            w.r.t the trainable variables as vector. This is mandatory
            for the LBFGS minimizer from scipy."""
            
            # Update weights in model
            set_weight_tensor(w)
            # Determine value of \phi and gradient w.r.t. \theta at w
            loss, grad = self.get_grad(X, u)
            
            # Store current loss for callback function            
            loss = loss.numpy().astype(np.float64)
            self.current_loss = loss            
            
            # Flatten gradient
            grad_flat = []
            for g in grad:
                grad_flat.extend(g.numpy().flatten())
            
            # Gradient list to array
            grad_flat = np.array(grad_flat,dtype=np.float64)
            
            # Return value and gradient of \phi as tuple
            return loss, grad_flat
        
        
        return scipy.optimize.minimize(fun=get_loss_and_grad,
                                       x0=x0,
                                       jac=True,
                                       method=method,
                                       callback=self.callback,
                                       **kwargs)
        
    def callback(self, xr=None):
        lambd = self.model.lambd.numpy()
        self.model.lambd_list.append(lambd)
        if self.iter % 500 == 0:
            tf.print('It {:05d}: loss = {:10.4e}, {}'.format(self.iter, self.current_loss, np.round(lambd, 3)))
        self.hist.append(self.current_loss)
        self.iter+=1        
        
    def plot_loss_history(self, ax=None):
        if not ax:
            fig = plt.figure(figsize=(7,5))
            ax = fig.add_subplot(111)
        ax.semilogy(range(len(self.hist)), self.hist,'k-')
        ax.set_xlabel('$n_{epoch}$')
        ax.set_ylabel('$\\phi^{n_{epoch}}$')
        return ax
    
# Initialize model
CMC = 1
model = PINN_NeuralNet(lb, ub)
model.build(input_shape=(None,1))

# Initilize PINN solver
solver = PINNSolver(model, X_r)

# Decide which optimizer should be used
mode = 'TFoptimizer'
# mode = 'ScipyOptimizer'

lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([200,5000],[1e-2,1e-3,5e-4])
# lr = 1e-3   #constant learning rate
N=5e4+1

try:
    runtime
except NameError:
    runtime = 0.

if mode == 'TFoptimizer':
    try:
        t0 = time()
        optim = tf.keras.optimizers.Adam(learning_rate = lr)
        solver.solve_with_TFoptimizer(optim, X_data, u_data, N=int(N))
        runtime += (time()-t0)/60.
        print('\nRuntime: {:.3f} minutes'.format(runtime))
    except KeyboardInterrupt:
        runtime += (time()-t0)/60.
        print('\nRuntime: {:.3f} minutes'.format(runtime))

elif mode == 'ScipyOptimizer':
    try:
        t0 = time()
        solver.solve_with_ScipyOptimizer(X_data, u_data,
                                        method='L-BFGS-B',
                                        options={'maxiter': 1000000,'maxfun': 1000000,'maxcor': 1000,
                                                              'maxls': 1000,'ftol' : 1.0 * np.finfo(float).eps})
        runtime += (time()-t0)/60.
        print('\nRuntime: {:.3f} minutes'.format(runtime))
    except KeyboardInterrupt:
        runtime += (time()-t0)/60.
        print('\nRuntime: {:.3f} minutes'.format(runtime))