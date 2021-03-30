#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 04:15:34 2020

@author: mohamedazizbhouri
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.utils.data
from torch.autograd import Variable

from SALib.analyze import morris
import seaborn as sns

import pickle

plt.close('all')
ind_plot = 0

torch.manual_seed(1234)
np.random.seed(seed=1)

torch.set_printoptions(precision=20)

class NeuralNet:
    # Initialize the class
    def __init__(self, X, Y, I0, hidden_dim, expli, trap, is_pred, batch_size, batch_county, Nm, Nc, Nt, tau, dt, Ntf):
        # Check if there is a GPU available
        if torch.cuda.is_available() == True:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor
        
        # Define PyTorch variables
        X = torch.from_numpy(X).type(self.dtype)
        Y = torch.from_numpy(Y).type(self.dtype)
        self.X = Variable(X, requires_grad=False)
        self.Y = Variable(Y, requires_grad=False)
        
        self.X_dim = X.shape[-1]
        self.Y_dim = 1 # Y.shape[-1]
        self.hidden_dim = hidden_dim
        self.tau = tau
        
        self.I0 = torch.from_numpy(I0).type(self.dtype)
        self.expli = expli.type(self.dtype)
        self.batch_size = batch_size
        self.batch_county = batch_county
        self.Nt = Nt
        self.dt = dt
        self.Ntf = Ntf
        self.trap = trap.type(self.dtype)
        self.is_pred = is_pred
        self.Nm = Nm
        self.Nc = Nc
        # Initialize model parameters
        self.U_f, self.b_f, self.W_f, \
        self.U_i, self.b_i, self.W_i, \
        self.U_s, self.b_s, self.W_s, \
        self.U_o, self.b_o, self.W_o, \
        self.V, self.c, self.E0 = self.initialize_LSTM()
        # Define optimizer
        self.optimizer = torch.optim.Adam([self.U_f, self.b_f,self. W_f, 
                                           self.U_i, self.b_i, self.W_i, 
                                           self.U_s, self.b_s, self.W_s, 
                                           self.U_o, self.b_o, self.W_o, 
                                           self.V, self.c, self.E0], lr=1e-3, weight_decay=0)
        # Logger
        self.loss_log = []
    
    # Initialize network weights and biases using Xavier initialization
    def initialize_LSTM(self):      
        # Xavier initialization
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = np.sqrt(2.0/(in_dim + out_dim))
            return Variable(xavier_stddev*torch.randn(in_dim, out_dim).type(self.dtype), requires_grad=True)
        
        # Forget Gate
        U_f = xavier_init(size=[self.X_dim, self.hidden_dim])
        b_f = Variable(torch.zeros(1,self.hidden_dim).type(self.dtype), requires_grad=True)
        W_f = Variable(torch.eye(self.hidden_dim).type(self.dtype), requires_grad=True)
        
        # Input Gate
        U_i = xavier_init(size=[self.X_dim, self.hidden_dim])
        b_i = Variable(torch.zeros(1,self.hidden_dim).type(self.dtype), requires_grad=True)
        W_i = Variable(torch.eye(self.hidden_dim).type(self.dtype), requires_grad=True)
        
        # Update Cell State
        U_s = xavier_init(size=[self.X_dim, self.hidden_dim])
        b_s = Variable(torch.zeros(1,self.hidden_dim).type(self.dtype), requires_grad=True)
        W_s = Variable(torch.eye(self.hidden_dim).type(self.dtype), requires_grad=True)
        
        # Ouput Gate
        U_o = xavier_init(size=[self.X_dim, self.hidden_dim])
        b_o = Variable(torch.zeros(1,self.hidden_dim).type(self.dtype), requires_grad=True)
        W_o = Variable(torch.eye(self.hidden_dim).type(self.dtype), requires_grad=True)
        
        # Ouput layer
        V = xavier_init(size=[self.hidden_dim, self.Y_dim])
        c = Variable(torch.zeros(1,self.Y_dim).type(self.dtype), requires_grad=True)
        
        E0 = Variable(self.inv_sigmoid_in(torch.rand(self.Nc,1,1)).type(self.dtype), requires_grad=True)
        
        return U_f, b_f, W_f, U_i, b_i, W_i, U_s, b_s, W_s, U_o, b_o, W_o, V, c, E0
    
    # Evaluates the forward pass
    def forward_pass(self, X, Nt):
        H = torch.zeros(X.shape[0], Nt, self.hidden_dim).type(self.dtype)
        S = torch.zeros(X.shape[0], Nt, self.hidden_dim).type(self.dtype)
        for i in range(0, self.tau):
            # Forget Gate
            FG = self.sigmoid_in(torch.matmul(H,self.W_f) + torch.matmul(X[:,i:i+Nt,:],self.U_f) + self.b_f)   
            # Input Gate
            IG = self.sigmoid_in(torch.matmul(H,self.W_i) + torch.matmul(X[:,i:i+Nt,:],self.U_i) + self.b_i)   
            # Update Cell State
            S_tilde = torch.tanh(torch.matmul(H,self.W_s) + torch.matmul(X[:,i:i+Nt,:],self.U_s) + self.b_s)
            S = FG*S + IG*S_tilde
            # Ouput Gate
            OG = self.sigmoid_in(torch.matmul(H,self.W_o) + torch.matmul(X[:,i:i+Nt,:],self.U_o) + self.b_o)
            H = OG*torch.tanh(S)
        # Ouput layer        
        H = 1.1*self.sigmoid_in( torch.matmul(H,self.V) + self.c )
        return H
    
    def sigmoid_in(self, x):
        return 1 / (1 + torch.exp(-x))
    def inv_sigmoid_in(self, x):
        return -torch.log(-1+1/x)

    # Evaluates predictions at test points    
    def predict(self, X_star, Nt):
        X_star = torch.from_numpy(X_star).type(self.dtype)
        y_star = self.forward_pass(X_star, Nt)
        y_star = y_star.cpu().data.numpy()
        return y_star

if __name__ == "__main__":     
    
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.size': 32,
                         'lines.linewidth': 4,
                         'axes.labelsize': 26,
                         'axes.titlesize': 26,
                         'xtick.labelsize': 18,
                         'ytick.labelsize': 18,
                         'legend.fontsize': 32,
                         'axes.linewidth': 4,
                        "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
                        "text.usetex": True,                # use LaTeX to write all text
                         })
    
    dpiv = 100
    figsize_val=(12.5,9.5)
    
    Nsim = 100 # Number of models
    
    gamma = 1/6.5 
    delta = 1/2.5 
    
    Nt = 81 # Number of days of the infections data
    
    Ntrain = 66 # Number of days of the infections data to be used for training
    
    Nm = 9 # Number of mobility and social behavior parameters
        
    Nc = 203 # Number of counties considered in the available data
    
    Ymin = 0.012 # Minimum value of the cumulative cases on the last day of the available data that needs to be satisfied by each county to be included in the individual plots generation
          
    hidden_dim = 120 # LSTM hidden state dimension
    
    tau = 21 # the time lag in days to consider between the mobility and social behavior parameters and the number of cumulative cases

    is_pred = 1
    
    # Load processed data
    X_train = np.load('processed/X_train.npy')
    Y_train = np.load('processed/Y_train.npy')         
    
    with open("processed/county_name.txt", "rb") as fp:
        county_name = pickle.load(fp)
       
    with open("processed/state_name.txt", "rb") as fp:
        state_name = pickle.load(fp)
    
    time = np.arange(0,Nt, 1)
    
    times = time/(Nt-1)
    dt = times[1]-times[0]
    t_train = times[:Ntrain]
    t_train = np.reshape(t_train, (-1, 1))
    
    ind_err = np.where(Y_train[:,-1]>Ymin)
    ind_err = ind_err[0]
    Nerr = sum(Y_train[:,-1]>Ymin)
    print('Number of big counties: ', Nerr)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    expli = torch.diag(-torch.ones(t_train.shape[0] - 2), diagonal = -1) + torch.eye(t_train.shape[0]-1)
    trap = torch.diag(torch.ones(t_train.shape[0] - 2), diagonal = -1)/2 + torch.eye(t_train.shape[0]-1)/2
        
    batch_size = t_train.shape[0] - 1
    batch_county = Nc
    
    if torch.cuda.is_available() == True:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    
    # Numpy arrays to save C(t) and \beta(t) predictions
    CC_list = np.zeros((Nsim,Nerr,Nt))
    beta_list = np.zeros((Nsim,Nerr,Nt))
    
    # Altered mobility simulation parameters
    ia = 130 # county index within loaded data 
    ma = 0 # Altered mobility index
    print(ia,county_name[ia],state_name[ia])
    
    # Alter mobility parameter ma for county ia
    day_a = 40
    X_train_a = np.array(X_train)
    X_train_a[ia,-day_a:,0] =  X_train_a[ia,-day_a:,0] + 0.4 
    
    #################################################################
    #################### altered_mobility plot ######################
    #################################################################
    plt.figure(figsize=figsize_val)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.plot(np.arange(0,Nt-1, 1), X_train[ia,tau:,ma], color='blue', label = "Baseline Trajectory")
    plt.plot(np.arange(0,Nt-1, 1), X_train_a[ia,tau:,ma], '--', color = "black", label = "Altered Trajectory")
    plt.xlim((0, Nt-1))
    plt.ylim(top = 0.1)
    # Activate below to plot mobility from day = -tau
#    plt.plot(np.arange(-tau,Nt-1, 1), X_train[ia,:,ma], 'b', label = "Baseline Trajectory")
#    plt.plot(np.arange(-tau,Nt-1, 1), X_train_a[ia,:,ma], '--', color = "black", label = "Altered Trajectory")
#    plt.xlim((-tau, Nt-1))
            
    plt.xlabel('$t \ (days)$',fontsize=36)
    yy = '$m_{' + str(ma+1) + '}(t)$'
    plt.ylabel(yy,fontsize=36)
    plt.legend(frameon=False,fontsize=32)
    if ma==0:
        plt.title("Retail and recreation change from baseline for " + county_name[ia] + " ; " + state_name[ia],fontsize=32)
    if ma==1:
        plt.title("Grocery and pharmacy mobility parameter for " + county_name[ia] + " ; " + state_name[ia],fontsize=32)
    if ma==5:
        plt.title("Residential mobility parameter for " + county_name[ia] + " ; " + state_name[ia],fontsize=32)
    tt = 'plots/altered_mobility.png'
    plt.savefig(tt, dpi = dpiv)  
    
    # Numpy arrays to save C(t) and \beta(t) predictions for altered and non-altered mobility parameter
    CC_na = np.zeros((Nsim,1,Nt))
    beta_na = np.zeros((Nsim,1,Nt))
    CC_a = np.zeros((Nsim,1,Nt))
    beta_a = np.zeros((Nsim,1,Nt))
    
    I0 = Y_train[:,0]
    I0 = I0[:,None,None]
    R0 = np.zeros((Nc,1,1))
      
    # Function to assign Weights to model
    def assign_params(Weights):
            
        model.U_f = Variable( torch.reshape( Weights[0:Nm*hidden_dim,:], (Nm, hidden_dim)) , requires_grad=True)        
        model.U_i = Variable( torch.reshape( Weights[Nm*hidden_dim:2*Nm*hidden_dim,:], (Nm, hidden_dim)) , requires_grad=True)     
        model.U_s = Variable( torch.reshape( Weights[2*Nm*hidden_dim:3*Nm*hidden_dim,:], (Nm, hidden_dim)) , requires_grad=True)     
        model.U_o = Variable( torch.reshape( Weights[3*Nm*hidden_dim:4*Nm*hidden_dim,:], (Nm, hidden_dim)) , requires_grad=True)     
            
        indf = 4*Nm*hidden_dim
        
        model.b_f = Variable( torch.reshape( Weights[ indf : indf+hidden_dim ,:], (1, hidden_dim)) , requires_grad=True)     
        model.b_i = Variable( torch.reshape( Weights[ indf+hidden_dim : indf+2*hidden_dim ,:], (1, hidden_dim)) , requires_grad=True)     
        model.b_s = Variable( torch.reshape( Weights[ indf+2*hidden_dim : indf+3*hidden_dim ,:], (1, hidden_dim)) , requires_grad=True)     
        model.b_o = Variable( torch.reshape( Weights[ indf+3*hidden_dim : indf+4*hidden_dim ,:], (1, hidden_dim)) , requires_grad=True)     
            
        indf = indf+4*hidden_dim
            
        model.W_f = Variable( torch.reshape( Weights[ indf : indf+hidden_dim**2,:], (hidden_dim, hidden_dim)) , requires_grad=True)     
        model.W_i = Variable( torch.reshape( Weights[ indf+hidden_dim**2 : indf+2*hidden_dim**2,:], (hidden_dim, hidden_dim)) , requires_grad=True)     
        model.W_s = Variable( torch.reshape( Weights[ indf+2*hidden_dim**2 : indf+3*hidden_dim**2,:], (hidden_dim, hidden_dim)) , requires_grad=True)     
        model.W_o = Variable( torch.reshape( Weights[ indf+3*hidden_dim**2 : indf+4*hidden_dim**2,:], (hidden_dim, hidden_dim)) , requires_grad=True)     
            
        indf = indf+4*hidden_dim**2
        
        model.V = Variable( Weights[indf:indf+hidden_dim,:] , requires_grad=True)     
        model.c = Variable( Weights[indf+hidden_dim:indf+hidden_dim+1,:] , requires_grad=True)     
        model.E0 = Variable( torch.reshape( Weights[indf+hidden_dim+1:,:], (Nc, 1, 1)) , requires_grad=True)     
           
    model = NeuralNet(X_train[:,0:tau+Ntrain-1,:], Y_train[:,:Ntrain], I0, hidden_dim, expli, trap, is_pred, batch_size, batch_county, Nm, Nc, Ntrain, tau, dt, Nt)
    
    def matrix_diag(diagonal):
        N = diagonal.shape[-1]
        shape = diagonal.shape[:-1] + (N, N)
        device, dtype = diagonal.device, diagonal.dtype
        result = torch.zeros(shape, dtype=dtype, device=device)
        indices = torch.arange(result.numel(), device=device).reshape(shape)
        indices = indices.diagonal(dim1=-2, dim2=-1)
        result.view(-1)[indices] = diagonal
        return result
        
    I0 = torch.from_numpy(I0).type(dtype)
    R0 = torch.from_numpy(R0).type(dtype)
        
    expli = torch.diag(-torch.ones(times.shape[0] - 2), diagonal = -1) + torch.eye(times.shape[0]-1)
    trap = torch.diag(torch.ones(times.shape[0] - 2), diagonal = -1)/2 + torch.eye(times.shape[0]-1)/2
    expli = expli.type(dtype)
    trap = trap.type(dtype)
        
    def Newton_system(Feval, Jac, x, eps,xprev,beta,betaprev,dt,gamma,delta):
        F_value = Feval(x,xprev,beta,betaprev,dt,gamma,delta)
        F_norm = np.linalg.norm(F_value)  # l2 norm of vector
            
        iteration_counter = 0
        while abs(F_norm) > eps and iteration_counter < 1000000:
            dx = np.linalg.solve(Jac(x,beta,dt,gamma,delta), -F_value)
            x = x + dx
            F_value = Feval(x,xprev,beta,betaprev,dt,gamma,delta)
            F_norm = np.linalg.norm(F_value)
            iteration_counter += 1
        
        if abs(F_norm) > eps:
            iteration_counter = -1
        return x, iteration_counter
    
    def Feval(x,xprev,beta,betaprev,dt,gamma,delta):
        return np.array([x[0]+dt/2*beta*x[0]*x[2] - ( xprev[0]-dt/2*betaprev*xprev[0]*xprev[2] ) ,
                         (1+delta*dt/2)*x[1]-dt/2*beta*x[0]*x[2] - ( (1-delta*dt/2)*xprev[1]+dt/2*betaprev*xprev[0]*xprev[2] ) ,
                         x[2]+gamma*dt/2*x[2]-dt/2*delta*x[1] - ( xprev[2]+dt/2*delta*xprev[1]-dt/2*gamma*xprev[2] ) ,
                         x[3]-dt/2*gamma*x[2] - ( xprev[3]+dt/2*gamma*xprev[2] ) ])
    
    def Jac(x,beta,dt,gamma,delta):
        return np.array([[1+dt/2*beta*x[2], 0, dt/2*beta*x[0], 0],
                         [-dt/2*beta*x[2], 1+delta*dt/2, -dt/2*beta*x[0], 0],
                         [0, -dt*delta/2, 1+gamma*dt/2, 0],
                         [0, 0, -dt/2*gamma, 1]])
        
    def myTrap(yinit,time,beta,dt,gamma,delta,epN):
        yo = np.zeros((time.shape[0],4))
        yo[0,:] = yinit
        for i in range(time.shape[0]-1):
            ytrial = np.array([yo[i,0]-dt*beta[i]*yo[i,0]*yo[i,2] ,
                               yo[i,1]+dt*beta[i]*yo[i,0]*yo[i,2]-dt*delta*yo[i,1] ,
                               yo[i,2]-gamma*dt*yo[i,2]+dt*delta*yo[i,1] ,
                               yo[i,3]+dt*gamma*yo[i,2] ])
                
            yo[i+1,:] , n = Newton_system(Feval, Jac, ytrial, epN, yo[i,:],beta[i+1],beta[i],dt,gamma,delta)
        return yo
    
    # Function to compute predictions and extrapolation in times
    def compute_sol(E0,I0,R0,X,Y):
        E0 = torch.from_numpy(E0).type(dtype)
        S0 = 1-E0-I0-R0
        expli = torch.diag(-torch.ones(t_train.shape[0] - 2), diagonal = -1) + torch.eye(t_train.shape[0]-1)
        trap = torch.diag(torch.ones(t_train.shape[0] - 2), diagonal = -1)/2 + torch.eye(t_train.shape[0]-1)/2
        expli = expli.type(dtype)
        trap = trap.type(dtype)
        Y = torch.from_numpy(Y).type(dtype)
        batch_county = X.shape[0]
        p0 = gamma
        p1 = delta
        Yl = Y[:,:Ntrain]
        bIA = torch.cat( ( (1-(Nt-1)*dt/2*p0)*I0 - Yl[:,0:1,None] , torch.zeros(batch_county,Ntrain-2,1).type(dtype) ), 1)
        IAm = torch.triangular_solve(bIA + torch.matmul( expli , Yl[:,1:,None] ), expli + (Nt-1)*dt*p0*trap, upper=False)[0]
        If = torch.cat( ( I0 , IAm ), 1)
        beta  = torch.from_numpy( model.predict(X[:,0:tau+Ntrain-1,:], Ntrain) ).type(dtype)
        bS = torch.cat( ( S0*( 1 - (Nt-1)*dt/2*beta[:,0:1,:]*If[:,0:1,:] ) , torch.zeros(batch_county,Ntrain-2,1).type(dtype) ), 1)
        betaIm = matrix_diag ( (beta[:,1:,:]*If[:,1:,:])[:,:,0] )
        betaIm1 = matrix_diag( (beta[:,1:-1,:]*If[:,1:-1,:])[:,:,0] )
        betaIm1f = torch.cat( (torch.zeros(batch_county,1,Ntrain-2).type(dtype),betaIm1) , 1 )
        betaIm1ff = torch.cat( (betaIm1f,torch.zeros(batch_county,Ntrain-1,1).type(dtype)) , 2 )
        Sm = torch.triangular_solve(bS, expli +(Nt-1)*dt/2*(betaIm+betaIm1ff), upper=False)[0]
        bE = torch.cat( ( E0 + (Nt-1)*dt/2*( S0*beta[:,0:1,:]*If[:,0:1,:] - p1*E0 ) , torch.zeros(batch_county,Ntrain-2,1).type(dtype) ), 1)
        betaSIm = beta[:,1:,:]*If[:,1:,:]*Sm
        Em = torch.triangular_solve(bE + (Nt-1)*dt*torch.matmul(trap,betaSIm), expli + (Nt-1)*dt*p1*trap, upper=False)[0]
        bI = torch.cat( ( If[:,0:1,:] + (Nt-1)*dt/2* (p1*E0-p0*If[:,0:1,:]) , torch.zeros(batch_county,Ntrain-2,1).type(dtype) ), 1)
        pred = torch.triangular_solve(bI + (Nt-1)*dt*p1*torch.matmul(trap,Em), expli + (Nt-1)*dt*p0*trap, upper=False)[0]
        bR = torch.cat( ( R0 + (Nt-1)*dt/2* p0*I0 , torch.zeros(batch_county,Ntrain-2,1).type(dtype) ), 1)
        Rm = torch.triangular_solve(bR + (Nt-1)*dt*p0*torch.matmul(trap,pred), expli, upper=False)[0]
        pred = torch.cat( ( I0 , pred) , 1)
        Rm = torch.cat( ( R0 , Rm) , 1)
        pred = pred.cpu().data.numpy()
        Rm = Rm.cpu().data.numpy()
        pred_MAP = np.zeros((batch_county,Nt,1))
        Rm_MAP = np.zeros((batch_county,Nt,1))

        for i in range(batch_county):
            yinit = np.array([Sm.cpu().data.numpy()[i,-1,0], Em.cpu().data.numpy()[i,-1,0], pred[i,-1,0], Rm[i,-1,0]])
            beta_pred = np.squeeze(model.predict(X[i:i+1,Ntrain-1:,:],Nt-Ntrain+1))
            epN = 1e-3
            Ypred = myTrap(yinit,time[Ntrain-1:],beta_pred,1,gamma,delta,epN)
                
            pred_MAP[i,:,0] = np.concatenate((pred[i,:,0],Ypred[1:,2]),axis=0)
            Rm_MAP[i,:,0] = np.concatenate((Rm[i,:,0],Ypred[1:,3]),axis=0)

        return pred_MAP + Rm_MAP
            
    Sic = np.zeros((Nc,Nm,tau))
    
    for s in range(Nsim):
        
        print('Sim :',s)
        
        yy = 'weights/weights_' + str(s) + '.npy'
        ww = np.load(yy)  
        ww = torch.from_numpy(ww).type(dtype)
        assign_params(ww)
        
        #################################################################
        ###################### Sensitivity Analysis #####################
        #################################################################
        
        # Define the model inputs
        bb = np.array([[X_train[:,0:tau+Ntrain-1,0].min(), X_train[:,0:tau+Ntrain-1,0].max()],
                       [X_train[:,0:tau+Ntrain-1,1].min(), X_train[:,0:tau+Ntrain-1,1].max()],
                       [X_train[:,0:tau+Ntrain-1,2].min(), X_train[:,0:tau+Ntrain-1,2].max()],
                       [X_train[:,0:tau+Ntrain-1,3].min(), X_train[:,0:tau+Ntrain-1,3].max()],
                       [X_train[:,0:tau+Ntrain-1,4].min(), X_train[:,0:tau+Ntrain-1,4].max()],
                       [X_train[:,0:tau+Ntrain-1,5].min(), X_train[:,0:tau+Ntrain-1,5].max()],
                       [X_train[:,0:tau+Ntrain-1,6].min(), X_train[:,0:tau+Ntrain-1,6].max()],
                       [X_train[:,0:tau+Ntrain-1,7].min(), X_train[:,0:tau+Ntrain-1,7].max()],
                       [X_train[:,0:tau+Ntrain-1,8].min(), X_train[:,0:tau+Ntrain-1,8].max()]])
        bb = np.tile(bb, (tau, 1))
        problem = {
            'num_vars': Nm*tau,
            'names': ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10',\
                      'x11','x12','x13','x14','x15','x16','x17','x18','x19','x20',\
                      'x21','x22','x23','x24','x25','x26','x27','x28','x29','x30',\
                      'x31','x32','x33','x34','x35','x36','x37','x38','x39','x40',\
                      'x41','x42','x43','x44','x45','x56','x47','x48','x49','x50',\
                      'x51','x52','x53','x54','x55','x56','x57','x58','x59','x60',\
                      'x61','x62','x63','x64','x65','x66','x67','x68','x69','x70',\
                      'x71','x72','x73','x74','x75','x76','x77','x78','x79','x80',\
                      'x81','x82','x83','x84','x85','x86','x87','x88','x89','x90',\
                      'x91','x92','x93','x94','x95','x96','x97','x98','x99','x100',\
                      'x101','x102','x103','x104','x105','x106','x107','x108','x109','x110',\
                      'x111','x112','x113','x114','x115','x116','x117','x118','x119','x120',\
                      'x121','x122','x123','x124','x125','x126','x127','x128','x129','x130',\
                      'x131','x132','x133','x134','x135','x136','x137','x138','x139','x140',\
                      'x141','x142','x143','x144','x145','x146','x147','x148','x149','x150',\
                      'x151','x152','x153','x154','x155','x156','x157','x158','x159','x160',\
                      'x161','x162','x163','x164','x165','x166','x167','x168','x169','x170',\
                      'x171','x172','x173','x174','x175','x176','x177','x178','x179','x180',\
                      'x181','x182','x183','x184','x185','x186','x187','x188','x189'],
            'bounds': bb,
            'num_trajectories': 1
        }
        #################################################################
        ####################### Method of Morris ########################
        #################################################################
        X = X_train[:,0:tau+Ntrain-1,:]
        beta = model.predict(X, Ntrain) # Nc x Nt x 1
        param_values_morris = np.zeros((Nc*Ntrain,tau*Nm))
        beta_morris = np.zeros((Nc*Ntrain))
        par = X[0,0:1,:]
        for i in range(tau-1):
            par = np.concatenate((par,X[0,i+1:i+2,:]),axis=1)
        param_values_morris[0,:] = par
        beta_morris[0] = beta[0,0]
        for j in range(Ntrain-1):
            par = X[0,j+1:j+2,:]
            for i in range(tau-1):
                par = np.concatenate((par,X[0,j+1+i+1:j+1+i+2,:]),axis=1)
            param_values_morris[j+1,:] = par
            beta_morris[j+1] = beta[0,j+1]
        for k in range(Nc-1):
            par = X[k+1,0:1,:]
            for i in range(tau-1):
                par = np.concatenate((par,X[k+1,i+1:i+2,:]),axis=1)
            param_values_morris[(k+1)*Ntrain+0,:] = par
            beta_morris[(k+1)*Ntrain+0] = beta[k+1,0]
            for j in range(Ntrain-1):
                par = X[k+1,j+1:j+2,:]
                for i in range(tau-1):
                    par = np.concatenate((par,X[k+1,j+1+i+1:j+1+i+2,:]),axis=1)
                param_values_morris[(k+1)*Ntrain+j+1,:] = par
                beta_morris[(k+1)*Ntrain+j+1] = beta[k+1,j+1]
        Ndata = (Nc*Ntrain//(tau*Nm+1))*(tau*Nm+1)
        indp = np.random.choice(Nc*Ntrain, Ndata, replace=False)
        param_values_morris = param_values_morris[indp,:]
        beta_morris = beta_morris[indp]
        Si = morris.analyze(problem, param_values_morris, beta_morris, conf_level=0.95,
                            print_to_console=False, num_levels=4)
        
        if s == 0:
            Si_avg = Si['mu_star']/sum(Si['mu_star'])
        else:
            Si_avg = Si_avg + Si['mu_star']/sum(Si['mu_star'])
        
        #################################################################
        ############### Compute C(t) & beta(t) with Extrap ##############
        #################################################################
    
        E0 = 3*I0.cpu().data.numpy()*sigmoid(model.E0.cpu().data.numpy())
        CC_list[s,:,:] = compute_sol(E0[ind_err,:,:],I0[ind_err,:,:],R0[ind_err,:,:],X_train[ind_err,:,:],Y_train[ind_err,:])[:,:,0]  # Nc x Nt x 1
        beta_list[s,:,:] = model.predict(X_train[ind_err,:,:], Nt)[:,:,0]
        
        CC_na[s,:,:] = compute_sol(E0[ia:ia+1,:,:],I0[ia:ia+1,:,:],R0[ia:ia+1,:,:],X_train[ia:ia+1,:,:],Y_train[ia:ia+1,:])[:,:,0] # Nc x Nt x 1
        beta_na[s,:,:] = model.predict(X_train[ia:ia+1,:,:], Nt)[:,:,0]
        
        CC_a[s,:,:] = compute_sol(E0[ia:ia+1,:,:],I0[ia:ia+1,:,:],R0[ia:ia+1,:,:],X_train_a[ia:ia+1,:,:],Y_train[ia:ia+1,:])[:,:,0] # Nc x Nt x 1
        beta_a[s,:,:] = model.predict(X_train_a[ia:ia+1,:,:], Nt)[:,:,0]
    
    #################################################################
    ################### Plot Sensitivty Analysis ####################
    #################################################################
    dat = Si_avg/Nsim
    
    dat = np.reshape(dat, (-1, Nm))
    dat = dat.T
    sns.set(font='serif',font_scale=3)
    plt.figure(figsize=(30,12))
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
       
    if Nm == 6:
        MobLab = ['Retail and Recreation','Grocery and Pharmacy','Parks','Transit Stations','Workplaces','Residential']
    if Nm == 9:
        MobLab = ['Retail and Recreation','Grocery and Pharmacy','Parks','Transit Stations','Workplaces','Residential','Distance Traveled','Non-Essential Visits','Human Encounters Rate']                 
    TauLab = np.arange(tau,0, -1)
    ax = sns.heatmap(dat,yticklabels=MobLab,xticklabels=TauLab.tolist(), cmap="RdBu_r")
    plt.xlabel('Time Lag (Days) : $j$',fontsize=32)
    plt.ylabel('Mobility : $i$',fontsize=32)
    plt.title('$s_{i,j}$: sensitivity of $\\beta$ to mobility parameters',fontsize=32)
    yy = 'plots/sens_avg.png'
    plt.savefig(yy, dpi = dpiv, bbox_inches = 'tight',pad_inches = 0.1)    
    
    #################################################################
    ############ Plot C(t), beta(t) & R0(t) with Extrap #############
    #################################################################
    
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.size': 32,
                         'lines.linewidth': 4,
                         'axes.labelsize': 26,
                         'axes.titlesize': 26,
                         'xtick.labelsize': 18,
                         'ytick.labelsize': 18,
                         'legend.fontsize': 32,
                         'axes.linewidth': 4,
                        "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
                        "text.usetex": True,                # use LaTeX to write all text
                         })
            
    iloc = -1
    for i in range(Nc):
        if Y_train[i,-1]>Ymin:
            iloc = iloc+1
            plt.figure(figsize=figsize_val)
            fig, ax1 = plt.subplots(figsize=figsize_val)
            plt.xticks(fontsize=32)
            plt.yticks(fontsize=32)
            
            mu_beta = np.mean(beta_list[:,iloc,:], axis = 0)
            sigma_beta = np.var(beta_list[:,iloc,:], axis = 0)
            
            lower_0 = mu_beta - 2.0*np.sqrt(sigma_beta)
            upper_0 = mu_beta + 2.0*np.sqrt(sigma_beta)
            
            plt.fill_between(time.flatten(), lower_0.flatten()/gamma, upper_0.flatten()/gamma, 
                     facecolor='grey', alpha=0.5, label='Predicted two std band of $R_0(t)$')
            plt.plot(time, mu_beta/gamma, 'k--', label = 'Predicted Average Trajectory of $R_0(t)$')
            
            plt.ylabel('$R_0(t)$',fontsize=36)
            
            ########################################################################
            
            plt.axvline(x=Ntrain-1, color='hotpink',linestyle =':' ,linewidth=6, label = 'Last day of Training data')
            plt.xlim((-1, Nt-1))
            plt.ylim(bottom=0)
            
            mm=1
            if i<138 and i>123:
                plt.axvline(x=0, color='orangered',linestyle =':' ,linewidth=6, label = '$\geq50$ gatherings ban')
                plt.axvline(x=6, color='limegreen',linestyle =':',linewidth=6 , label = 'Stay at home')
            if i==108 or i==115:
                plt.axvline(x=0, color='orangered',linestyle =':' ,linewidth=6, label = '$\geq50$ gatherings ban')
                plt.axvline(x=5, color='limegreen',linestyle =':',linewidth=6 , label = 'Stay at home')
            if i==83 or i==86:
                plt.axvline(x=1, color='orangered',linestyle =':' ,linewidth=6, label = '$\geq50$ gatherings ban')
                plt.axvline(x=8, color='limegreen',linestyle =':',linewidth=6 , label = 'Stay at home')
            if i==33:
                plt.axvline(x=9, color='orangered',linestyle =':' ,linewidth=6, label = '$\geq50$ gatherings ban')
                plt.axvline(x=16, color='limegreen',linestyle =':',linewidth=6 , label = 'Stay at home')
            if i==55:
                plt.axvline(x=2, color='orangered',linestyle =':' ,linewidth=6, label = '$\geq50$ gatherings ban')
                plt.axvline(x=5, color='limegreen',linestyle =':',linewidth=6 , label = 'Stay at home')
            if i==2:
                plt.axvline(x=10, color='limegreen',linestyle =':',linewidth=6 , label = 'Stay at thome')
            if i==29:
                plt.axvline(x=5, color='orangered',linestyle ='--' ,linewidth=6, label = '$\geq50$ gatherings ban')
                xx = 5*np.ones_like(np.arange(mm/40,mm-mm/40,mm/20))
                plt.axvline(x=5, color='limegreen',linestyle =':',linewidth=6 , label = 'Stay at home')
            if i==69:
                plt.axvline(x=1, color='orangered',linestyle =':' ,linewidth=6, label = '$\geq50$ gatherings ban')
                plt.axvline(x=7, color='limegreen',linestyle =':',linewidth=6 , label = 'Stay at home')
            if i==78:
                plt.axvline(x=0, color='orangered',linestyle =':' ,linewidth=6, label = '$\geq50$ gatherings ban')
                plt.axvline(x=14, color='limegreen',linestyle =':',linewidth=6 , label = 'Stay at home')
            if i==163:
                plt.axvline(x=6, color='orangered',linestyle ='--' ,linewidth=6, label = '$\geq50$ gatherings ban')
                plt.axvline(x=6, color='limegreen',linestyle =':',linewidth=6 , label = 'Stay at home')
            if i==192:
                plt.axvline(x=5, color='orangered',linestyle =':' ,linewidth=6, label = '$\geq50$ gatherings ban')
                plt.axvline(x=11, color='limegreen',linestyle =':',linewidth=6 , label = 'Stay at home')
            
            plt.legend(loc='upper center', bbox_to_anchor=(0.0,-0.2))
            
            if county_name[i]=='District of Columbia':
                plt.title(county_name[i],fontsize=36)
            else:
                plt.title(county_name[i] + " ; " + state_name[i],fontsize=36)
            plt.xlabel('$t \ (days)$',fontsize=36)
            
            ########################################################################
            
            ax2 = ax1.twinx()
            ax2.tick_params(labelsize=32,color='b',labelcolor='b')
            
            ax1.tick_params(length=6, width=2)
            ax2.tick_params(length=6, width=2)
            
            mu_CC = np.mean(CC_list[:,iloc,:], axis = 0)
            sigma_CC = np.var(CC_list[:,iloc,:], axis = 0)
            
            lower_0 = mu_CC - 2.0*np.sqrt(sigma_CC)
            upper_0 = mu_CC + 2.0*np.sqrt(sigma_CC)
            
            ax2.fill_between(time.flatten(), 100 * lower_0.flatten(), 100 * upper_0.flatten(), 
                     facecolor='orange', alpha=0.5, label='Predicted two std band of $C(t)$')
            
            ax2.plot(time, 100 * mu_CC ,'--', color = 'blue' , label = 'Predicted Average Trajectory of $C(t)$')
            ax2.plot(time, 100 * Y_train[i,:], '-', color = 'red', label = 'True Trajectory of $C(t)$')
            
            ax2.set_ylabel('$C(t) \ (\%)$',fontsize=36,color='b')
            ax2.set_ylim(bottom=0)
#            plt.legend(frameon=False, prop={'size': 20})
            
            plt.legend(loc='upper center', bbox_to_anchor=(1.0,-0.2))
            tt = 'plots/C_R0_' + county_name[i] + "_" + state_name[i] + ".png"
            
            plt.savefig(tt, dpi = dpiv, bbox_inches='tight') 
    
    #################################################################
    ############ Plot C(t), beta(t) & R0(t) with Extrap  ############
    ############### for 1 County and altered mobility ###############
    #################################################################
    
    plt.figure(figsize=figsize_val)   
    fig, ax1 = plt.subplots(figsize=figsize_val)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)    
    
    mu_CC_na = np.mean(CC_na[:,0,:], axis = 0)
    sigma_CC_na = np.var(CC_na[:,0,:], axis = 0)
    lower_0_na = mu_CC_na - 2.0*np.sqrt(sigma_CC_na)
    upper_0_na = mu_CC_na + 2.0*np.sqrt(sigma_CC_na)
    plt.fill_between(time.flatten(), 100 * lower_0_na.flatten(), 100 * upper_0_na.flatten(), 
                     facecolor='orange', alpha=0.5, label="Predicted two std band of Baseline Trajectory")     
    plt.plot(time, 100 * mu_CC_na, '--', color='blue', label = "Average Prediction of Baseline Trajectory")
    plt.plot(time, 100 * Y_train[ia,:], 'r-', label = "True Trajectory")    
    
    mu_CC_a = np.mean(CC_a[:,0,:], axis = 0)
    sigma_CC_a = np.var(CC_a[:,0,:], axis = 0)
    lower_0_a = mu_CC_a - 2.0*np.sqrt(sigma_CC_a)
    upper_0_a = mu_CC_a + 2.0*np.sqrt(sigma_CC_a)
    plt.fill_between(time.flatten(), 100 * lower_0_a.flatten(), 100 * upper_0_a.flatten(), 
                     facecolor='grey', alpha=0.5, label="Predicted two std band of Altered Trajectory")
    plt.plot(time, 100 * mu_CC_a, 'k--', label = "Average Prediction of Altered Trajectory")
    
    plt.axvline(x=Nt-day_a-1, color='k',linestyle =':' ,linewidth=6, label = 'First date of altered data') # reactivate if change for last days
    plt.xlim((0, Nt-1))
    plt.ylim(bottom=0)
    plt.xlabel('$t \ (days)$',fontsize=36)
    plt.ylabel('$C(t) \ (\%)$',fontsize=36)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.2))
    plt.savefig("plots/altered_mobility_C.png", dpi = dpiv, bbox_inches='tight')  
    
    
    plt.figure(figsize=figsize_val)
    fig, ax1 = plt.subplots(figsize=figsize_val)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    
    mu_beta_na = np.mean(beta_na[:,0,:], axis = 0)
    sigma_beta_na = np.var(beta_na[:,0,:], axis = 0)
    lower_0_na = mu_beta_na - 2.0*np.sqrt(sigma_beta_na)
    upper_0_na = mu_beta_na + 2.0*np.sqrt(sigma_beta_na)
    plt.fill_between(time.flatten(), lower_0_na.flatten(), upper_0_na.flatten(),
                     facecolor='orange', alpha=0.5, label="Predicted two std band of Baseline Trajectory")
    plt.plot(time, mu_beta_na, '--', color='blue', label = "Average Prediction of Baseline Trajectory")
       
    mu_beta_a = np.mean(beta_a[:,0,:], axis = 0)
    sigma_beta_a = np.var(beta_a[:,0,:], axis = 0)
    lower_0_a = mu_beta_a - 2.0*np.sqrt(sigma_beta_a)
    upper_0_a = mu_beta_a + 2.0*np.sqrt(sigma_beta_a)
    plt.fill_between(time.flatten(), lower_0_a.flatten(), upper_0_a.flatten(),
                     facecolor='grey', alpha=0.5, label="Predicted two std band of Altered Trajectory")
    plt.plot(time, mu_beta_a, 'k--', label = "Average Prediction of Altered Trajectory")
          
    plt.axvline(x=Nt-day_a-1, color='k',linestyle =':' ,linewidth=6, label = 'First date of altered data') # reactivate if change for last days
    plt.xlim((0, Nt-1))
    plt.ylim(bottom=0)
    plt.xlabel('$t \ (days)$',fontsize=36)
    plt.ylabel('$\\beta(t)$',fontsize=36)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.2))
    plt.savefig("plots/altered_mobility_beta.png", dpi = dpiv, bbox_inches='tight')  
         
    
    plt.figure(figsize=figsize_val)
    fig, ax1 = plt.subplots(figsize=figsize_val)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    
    plt.fill_between(time.flatten(), lower_0_na.flatten()/gamma, upper_0_na.flatten()/gamma,
                     facecolor='orange', alpha=0.5, label="Predicted two std band of Baseline Trajectory")
    plt.plot(time, mu_beta_na/gamma, '--', color='blue', label = "Average Prediction of Baseline Trajectory")
    plt.fill_between(time.flatten(), lower_0_a.flatten()/gamma, upper_0_a.flatten()/gamma,
                     facecolor='grey', alpha=0.5, label="Predicted two std band of Altered Trajectory")
    plt.plot(time, mu_beta_a/gamma, 'k--', label = "Average Prediction of Altered Trajectory")
    
    plt.axvline(x=Nt-day_a-1, color='k',linestyle =':' ,linewidth=6, label = 'First date of altered data') # reactivate if change for last days
    plt.xlim((0, Nt-1))
    plt.ylim(bottom=0)
    plt.xlabel('$t \ (days)$',fontsize=36)
    plt.ylabel('$R_0(t)$',fontsize=36)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.2))
    plt.savefig("plots/altered_mobility_R0.png", dpi = dpiv, bbox_inches='tight')
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    