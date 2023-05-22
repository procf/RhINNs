import numpy as np
import tensorflow as tf
from scipy.special import gamma, factorial
import sys
import random
import os

DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)
SEED=42
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
log10 = tf.experimental.numpy.log10

def Mittag_Leffler(t_d, kappa, mu):
    E = 0.0
    for n in range(0,100):
        E += t_d**n/gamma(kappa*n + mu)
    return E

def generate_data(tmin,tmax,N_d,N_r,alpha,beta,E,tau):
    #Data points
    t_d = np.linspace(tmin, tmax, N_d, dtype=DTYPE)
    kappa = alpha - beta
    mu = 1. + alpha
    J = 1. / E * (t_d/tau)**alpha * Mittag_Leffler(-(t_d/tau)**kappa, kappa, mu)
    X_data = tf.reshape(tf.convert_to_tensor(t_d, dtype=DTYPE),(-1,1))
    u_data = tf.reshape(tf.convert_to_tensor(J, dtype=DTYPE),(-1,1))
    # Residual points
    t_r = np.linspace(tmin, tmax, N_d, dtype=DTYPE)
    h = t_r[1] - t_r[0]
    X_r = tf.reshape(tf.convert_to_tensor(t_r, dtype=DTYPE),(-1,1))
    return X_data, u_data, X_r, h

class PINN_NeuralNet(tf.keras.Model):
    def __init__(self, 
            output_dim=1,
            num_hidden_layers=4, 
            num_neurons_per_layer=10,
            activation='tanh',
            kernel_initializer='glorot_normal',
            **kwargs):
        super().__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim
        self.lambd = tf.Variable(self.CM()[0]*tf.ones(self.CM()[1]), trainable=True, dtype=DTYPE,
                                 constraint=lambda x: tf.clip_by_value(x, self.CM()[2], self.CM()[3]))
        
        self.hidden = [tf.keras.layers.Dense(num_neurons_per_layer,
                             activation=tf.keras.activations.get(activation),
                             kernel_initializer=kernel_initializer)
                           for _ in range(self.num_hidden_layers)]
        self.out = tf.keras.layers.Dense(output_dim)

    def CM(self):
        num_param = 2
        init = [.45, .3]
        low = [1e-2, 1e-2]
        high = [.99, .99]
        return init, num_param, low, high
    
    def call(self, X):
        Z = X
        for i in range(self.num_hidden_layers):
            Z = self.hidden[i](Z)
        return self.out(Z)

class PINNSolver():
    def __init__(self, model, X_r):
        self.model = model
        self.t = X_r
        self.hist = []
        self.iter = 0

    def Caputo_coeff(self,alpha,k,data_point):
        if k==0:
            return 1.
        elif k==data_point:
            return (k-1.)**(1.-alpha) - k**(1.-alpha) + (1.-alpha) * k**(-alpha)
        else:
            return (k-1.)**(1.-alpha) - 2*k**(1.-alpha) + (k+1)**(1.-alpha)

    def Caputo(self, alpha, f):
        C1 = [self.Caputo_coeff(alpha, index, index) for index in range(N_r)]
        C2 = [self.Caputo_coeff(alpha, index, N_r-1) for index in range(N_r)]
        frac1 = [tf.math.reduce_sum([f[i-j]*C2[j] for j in range(i)])+f[0]*C1[i] for i in range(N_r)]
        var1 = h**(-alpha) * tf.math.exp(-tf.math.lgamma(2.-alpha))
        fractional = var1 * frac1
        return fractional

    def get_r(self):     
        u = self.model(self.t)
        alpha, beta = [self.model.lambd[j] for j in range(self.model.CM()[1])]
        frac1 = self.Caputo(alpha,u)
        frac2 = self.Caputo(beta,u)
        res = E * tau**alpha * frac1 + E * tau**beta * frac2 - 1.
        return res
   
    def loss_fn(self, X, u):       
        r = self.get_r()
        Loss_eq = tf.reduce_mean(tf.square(r))
        y_pred = self.model(X)
        Loss_data = tf.reduce_mean(tf.square(u - y_pred))
        return Loss_eq + Loss_data 
    
    def get_grad(self, X, u):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_variables)
            loss = self.loss_fn(X, u)            
        g = tape.gradient(loss, self.model.trainable_variables)
        del tape       
        return loss, g
    
    def solve_with_TFoptimizer(self, optimizer, X, u, N=1001):
        @tf.function
        def train_step():
            loss, grad_theta = self.get_grad(X, u)           
            optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            return loss        
        for i in range(N):           
            loss = train_step()            
            self.current_loss = loss.numpy()
            self.callback()
 
    def callback(self, xr=None):
        lambd = self.model.lambd.numpy()
        if self.iter % 1000 == 0:
            tf.print('It {:05d}: loss = {:10.4e}, {}'.format(self.iter, self.current_loss, np.round(lambd, 3)),output_stream=sys.stdout)
        self.hist.append(self.current_loss)
        self.iter+=1

if __name__ == '__main__':
    alpha = float(sys.argv[1])
    beta = float(sys.argv[2])
    tmin, tmax, N_d, N_r = 0.001, 5., 100, 100
    E, tau = 1., 20.
    X_data, u_data, X_r, h = generate_data(tmin,tmax,N_d,N_r,alpha,beta,E,tau)
    del alpha, beta

    model = PINN_NeuralNet(output_dim=1,num_hidden_layers=4,num_neurons_per_layer=10)
    model.build(input_shape=(None,1))
    solver = PINNSolver(model, X_r)
    
    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([200,1000],[1e-2,1e-3,5e-4])
    optim = tf.keras.optimizers.Adam(learning_rate=lr)
    solver.solve_with_TFoptimizer(optim, X_data, u_data, N=100001)
