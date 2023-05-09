from time import time
from datetime import datetime
import os
import sys
import math
import tensorflow as tf
import numpy as np
import scipy.optimize
import pandas as pd
from numpy import random
import itertools
from scipy.integrate import odeint

DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)
SEED=42
os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

#class to set basic architecture of the PINN model
class PINN_NeuralNet(tf.keras.Model):
    def __init__(self,
            output_dim=2,
            num_hidden_layers=4,
            num_neurons_per_layer=25,
            activation='tanh',
            kernel_initializer='glorot_normal',
            CMC=1,
            **kwargs):
        super().__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim
        self.CMC = CMC
        self.lambd = tf.Variable(self.CM()[0]*tf.ones(self.CM()[1]), trainable=True, dtype=DTYPE,
                                 constraint=lambda x: tf.clip_by_value(x, self.CM()[2], self.CM()[3]))
        self.lambd_list = []

        self.hidden = [tf.keras.layers.Dense(num_neurons_per_layer,
                             activation=tf.keras.activations.get(activation),
                             kernel_initializer=kernel_initializer)
                           for _ in range(self.num_hidden_layers)]
        self.out = tf.keras.layers.Dense(output_dim)

    def CM(self):
        if self.CMC == 1: #TEVP
            num_param = 6
            init = [1., 1., 1., 1., 1., 1.]
            low = [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4]
            high = [np.infty, np.infty, np.infty, np.infty, 2.0, 2.0]
            return init, num_param, low, high

    def call(self, X):
        Z = X
        for i in range(self.num_hidden_layers):
            Z = self.hidden[i](Z)
        return self.out(Z)

#class to train the PINN model
class PINNSolver():
    def __init__(self, model, X_f, X_f0, smax):
        self.model = model
        self.x1_f = tf.gather(X_f,[0],axis=1)
        self.x2_f = tf.gather(X_f,[1],axis=1)
        self.x3_f = tf.gather(X_f,[2],axis=1)
        self.x1_f0 = tf.gather(X_f0,[0],axis=1)
        self.x2_f0 = tf.gather(X_f0,[1],axis=1)
        self.x3_f0 = tf.gather(X_f0,[2],axis=1)
        self.smax = smax
        self.hist = []
        self.iter = 0

    def get_r(self):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.x1_f)
            y_pred = self.model(tf.concat([self.x1_f,self.x2_f,self.x3_f],axis=1))
            y2_pred = tf.gather(y_pred, [1], axis=1)
            y1_pred = tf.gather(y_pred, [0], axis=1)
        y2_t = tape.gradient(y2_pred, self.x1_f)
        y1_t = tape.gradient(y1_pred, self.x1_f)
        del tape
        G, sy, es, ep, kp, kn = [self.model.lambd[j] for j in range(self.model.CM()[1])]
        gdot = self.x2_f*self.x3_f*tf.math.cos(self.x3_f*self.x1_f)
        R1 = y1_t - (G/(es + ep))*(-y1_pred + sy*y2_pred/self.smax + (es + ep*y2_pred)*gdot/self.smax)
        R2 = y2_t - (kp*(1. - y2_pred) - kn*y2_pred*tf.math.abs(gdot))
        return R1, R2

    def loss_fn(self, X_data, X_f0, y_data):
        R1, R2 = self.get_r()
        Loss_eq = tf.reduce_mean(tf.square(R1)) + tf.reduce_mean(tf.square(R2))
        y_pred = self.model(X_data)
        y_init = self.model(X_f0)
        Loss_init = tf.reduce_mean(tf.square(tf.gather(y_init, [0], axis=1) - 0.0))\
         + tf.reduce_mean(tf.square(tf.gather(y_init, [1], axis=1) - 1.))
        Loss_data = tf.reduce_mean(tf.square(y_data - tf.gather(y_pred, [0], axis=1)))
        loss = Loss_data + Loss_eq + Loss_init
        loss_frac = [Loss_eq, Loss_init, Loss_data]
        return loss, loss_frac

    def get_grad(self, X_data, X_f0, y_data):
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            loss, loss_frac = self.loss_fn(X_data, X_f0, y_data)
        g = tape.gradient(loss, self.model.trainable_variables)
        del tape
        return loss, g, loss_frac

    def solve_with_TFoptimizer(self, optimizer, X_data, X_f0, y_data, N=1001):
        @tf.function
        def train_step():
            loss, grad_theta, loss_frac = self.get_grad(X_data, X_f0, y_data)
            optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            return loss, loss_frac
        for i in range(N):
            loss, loss_frac = train_step()
            self.loss_frac = loss_frac
            self.current_loss = loss.numpy()
            self.callback()

    def callback(self, xr=None):
        lambd = self.model.lambd.numpy()
        self.model.lambd_list.append(lambd)
        if self.iter % 5000 == 0:
            tf.print(self.iter,self.current_loss,lambd[0],lambd[1],lambd[2],lambd[3],lambd[4],lambd[5])
        self.iter+=1

#function to solve TEVP constitutive ODEs
def TEVP(params, smax, tmin, tmax, sa, omega, N_d_ode):
    G, sy, es, ep, kp, kn = np.array(params)
    def odes(x, t):
        S = x[0]
        L = x[1]
        dSdt = G/(es+ep)*(-S + sy*L/smax + (es + ep*L)*sa*np.cos(omega*t)*omega/smax)
        dLdt = kp*(1 - L) - kn*L*sa*np.abs(np.cos(omega*t))*omega
        return [dSdt, dLdt]
    x0 = [0.0, 1.]
    t = np.logspace(np.log10(tmin),np.log10(tmax), N_d_ode)
    x = odeint(odes, x0 ,t)
    SS = x[:,0]
    L = x[:,1]
    SA = sa*np.ones(N_d_ode)
    W = omega*np.ones(N_d_ode)
    return t, SA, W, SS, L

def generate_date(N_g,gmin,gmax,N_o,Omin,Omax,tmin,tmax,Smax,N_d):
    Time, StrainAmplitude, Frequency, ShearStress, Lambda = [], [], [], [], []
    strain_amp = np.linspace(gmin, gmax, N_g)
    omega = np.linspace(Omin,Omax,N_o)
    param_exact = [40., 10., 10., 5, 0.1, 0.3] #G, sy, es, ep, kp, kn
    for ome in omega:
        for sa in strain_amp:
            t, SA, W, SS, L = TEVP(param_exact, Smax, tmin, tmax, sa, ome, N_d)
            Time = np.append(Time, t, axis=None)
            StrainAmplitude = np.append(StrainAmplitude, SA, axis=None)
            Frequency = np.append(Frequency, W, axis=None)
            ShearStress = np.append(ShearStress, SS, axis=None)
            Lambda = np.append(Lambda, L, axis=None)
    df = pd.DataFrame({"Time" : Time, "StrainAmplitude" : StrainAmplitude, "Frequency" : Frequency, "ShearStress" : ShearStress, "Lambda" : Lambda})
    df.to_excel("StartUp.xlsx", index=False)

def read_data():
    df = pd.read_excel('StartUp.xlsx')
    x1_d = tf.reshape(tf.convert_to_tensor(df['Time'], dtype=DTYPE),(-1,1))
    x2_d = tf.reshape(tf.convert_to_tensor(df['StrainAmplitude'], dtype=DTYPE), (-1,1))
    x3_d = tf.reshape(tf.convert_to_tensor(df['Frequency'], dtype=DTYPE), (-1,1))
    y1_d = tf.reshape(tf.convert_to_tensor(df['ShearStress'], dtype=DTYPE), (-1,1))
    y2_d = tf.reshape(tf.convert_to_tensor(df['Lambda'], dtype=DTYPE), (-1,1))
    X_data = tf.concat([x1_d, x2_d, x3_d], axis=1)
    y_data = tf.concat([y1_d], axis=1)

    tmin, tmax = np.min(x1_d), np.max(x1_d)
    xmin, xmax = np.min(x2_d), np.max(x2_d)
    ymin, ymax = np.min(x3_d), np.max(x3_d)
    lb = tf.constant([tmin, xmin, ymin], dtype=DTYPE).numpy()
    ub = tf.constant([tmax, xmax, ymax], dtype=DTYPE).numpy()
    
    #Collocation points
    N_res=500
    t_f = np.logspace(np.log10(tmin), np.log10(tmax), N_res)
    x_f = np.linspace(lb[1], ub[1],4)
    y_f = np.linspace(lb[2], ub[2],4)
    X_f = list(itertools.product(t_f, x_f, y_f))
    X_f = tf.convert_to_tensor(X_f,dtype=DTYPE)
    
    #Initial points
    N_i = 50
    t_f0=np.asarray([tmin])
    x_f0 = np.linspace(lb[1], ub[1],50)
    y_f0 = np.linspace(lb[2], ub[2],4)
    X_f0 = list(itertools.product(t_f0, x_f0, y_f0))
    X_f0 = tf.convert_to_tensor(X_f0,dtype=DTYPE)
    return lb, ub, X_f, X_f0, X_data, y_data

if __name__ == '__main__':
    Omin, Omax, N_o = 1., 5.,1
    gmin, gmax, N_g = .1, 1., 3
    Smax = 20.8477
    N_d = 200
    tmin, tmax = 0.001, 3.0*2.0*np.pi/Omax
    in_dim, out_dim = 3, 2
    generate_date(N_g,gmin,gmax,N_o,Omin,Omax,tmin,tmax,Smax,N_d)
    lb, ub, X_f, X_f0, X_data, y_data = read_data()
    model = PINN_NeuralNet(output_dim=out_dim,num_hidden_layers=4,num_neurons_per_layer=20,CMC=1)
    model.build(input_shape=(None,in_dim))
    solver = PINNSolver(model, X_f, X_f0, Smax)
    
    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([20000,50000],[1e-2,2e-3,1e-3])
    optim = tf.keras.optimizers.Adam(learning_rate = lr)
    solver.solve_with_TFoptimizer(optim, X_data, X_f0, y_data, N=int(10e5))
