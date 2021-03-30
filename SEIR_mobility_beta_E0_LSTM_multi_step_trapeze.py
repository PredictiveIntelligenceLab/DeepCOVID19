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
import timeit

plt.close('all')
ind_plot = 0

torch.manual_seed(1234)
np.random.seed(seed=1234)

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
        self.Y_dim = 1 
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
    
    # Create diagonal matrices from a batch of vectors
    def matrix_diag(self, diagonal):
        N = diagonal.shape[-1]
        shape = diagonal.shape[:-1] + (N, N)
        device, dtype = diagonal.device, diagonal.dtype
        result = torch.zeros(shape, dtype=dtype, device=device)
        indices = torch.arange(result.numel(), device=device).reshape(shape)
        indices = indices.diagonal(dim1=-2, dim2=-1)
        result.view(-1)[indices] = diagonal
        return result
    
    # Computes the mean square error loss
    def compute_loss(self, X, Y, I0, idx):
        ep = 1e-7 # to be used if a normalized loss is considered (check the variable "res")
        
        p0 = gamma
        p1 = delta
        
        E0l = 3*I0*self.sigmoid_in(self.E0[idx,:,:])
        S0 = 1-I0-E0l
        
        # compute I(t) from R(t)+I(t)
        bIA = torch.cat( ( (1-(self.Ntf-1)*self.dt/2*p0)*I0 - Y[:,0:1,None] , torch.zeros(self.batch_county,self.Nt-2,1).type(self.dtype) ), 1)
        
        IAm = torch.triangular_solve(bIA + torch.matmul( self.expli , Y[:,1:,None] ), self.expli + (self.Ntf-1)*self.dt*p0*self.trap, upper=False)[0]
        If = torch.cat( ( I0 , IAm ), 1)

        beta  = self.forward_pass(X, self.Nt)
        
        # Build RHS of S equation
        bS = torch.cat( ( S0*( 1 - (self.Ntf-1)*self.dt/2*beta[:,0:1,:]*If[:,0:1,:] ) , torch.zeros(self.batch_county,self.Nt-2,1).type(self.dtype) ), 1) # size: batch_county x Nt-1 x 1 
        # Build matrix of S equation
        betaIm = self.matrix_diag ( (beta[:,1:,:]*If[:,1:,:])[:,:,0] ) # size: batch_county x Nt-1 x Nt-1
        betaIm1 = self.matrix_diag( (beta[:,1:-1,:]*If[:,1:-1,:])[:,:,0] ) # size: batch_county x Nt-2 x Nt-2
        betaIm1f = torch.cat( (torch.zeros(self.batch_county,1,self.Nt-2).type(self.dtype),betaIm1) , 1 ) # size: batch_county x Nt-1 x Nt-2
        betaIm1ff = torch.cat( (betaIm1f,torch.zeros(self.batch_county,self.Nt-1,1).type(self.dtype)) , 2 ) # size: batch_county x Nt-1 x Nt-1
        # S prediction
        Sm = torch.triangular_solve(bS, self.expli +(self.Ntf-1)*self.dt/2*(betaIm+betaIm1ff), upper=False)[0] # size: batch_county x Nt-1 x 1 
        
        # Build RHS of E equation
        bE = torch.cat( ( E0l + (self.Ntf-1)*self.dt/2*( S0*beta[:,0:1,:]*If[:,0:1,:] - p1*E0l ) , torch.zeros(self.batch_county,self.Nt-2,1).type(self.dtype) ), 1) # size: batch_county x Nt-1 x 1 
        betaSIm = beta[:,1:,:]*If[:,1:,:]*Sm # size: batch_county x Nt-1 x 1 
        # E prediction
        Em = torch.triangular_solve(bE + (self.Ntf-1)*self.dt*torch.matmul(self.trap,betaSIm), self.expli + (self.Ntf-1)*self.dt*p1*self.trap, upper=False)[0] # size: batch_county x Nt-1 x 1 
        
        # Build RHS of I equation
        bI = torch.cat( ( If[:,0:1,:] + (self.Ntf-1)*self.dt/2* (p1*E0l-p0*If[:,0:1,:]) , torch.zeros(self.batch_county,self.Nt-2,1).type(self.dtype) ), 1)
        if self.is_pred == 1:
            # I prediction
            pred = torch.triangular_solve(bI + (self.Ntf-1)*self.dt*p1*torch.matmul(self.trap,Em), self.expli + (self.Ntf-1)*self.dt*p0*self.trap, upper=False)[0] # size: batch_county x Nt-1 x 1 
            res = (If[:,1:,:] - pred) #/ (torch.mean(If[:,1:,0],dim=1)[:,None,None] + ep) # size: batch_county x Nt-1 x 1 
        else:
            # I equation residual
            res = ( torch.matmul( (self.expli + (self.Ntf-1)*self.dt*p0*self.trap) , If[:,1:,:] ) - ( bI + (self.Ntf-1)*self.dt*p1*torch.matmul(self.trap,Em) ) ) #/ (torch.mean(If[:,1:,0]-If[:,:-1,0],dim=1)[:,None,None] + ep) # size: batch_county x Nt-1 x 1    
            
        loss = torch.mean(res[:,torch.randperm(self.Nt-1)[0:self.batch_size],:]**2)    
        
        return loss # / (X.shape[0]-1)

    # Fetches a mini-batch of data
    def fetch_minibatch(self,X, y, I0):
        idx = torch.randperm(self.Nc)[0:self.batch_county]
        X_batch = X[idx,:,:]
        y_batch = y[idx,:]
        I0_batch = I0[idx,:,:]       
        return X_batch, y_batch, I0_batch, idx
    
    # Trains the model by minimizing the MSE loss
    def train(self, nIter = 10000):
        start_time = timeit.default_timer()
        for it in range(nIter):
            # Fetch mini-batch
            X_batch, Y_batch, I0_batch, idx = self.fetch_minibatch(self.X, self.Y, self.I0)
            loss = self.compute_loss(X_batch, Y_batch, I0_batch, idx)
            # Backward pass
            loss.backward()
            
            # update parameters
            self.optimizer.step()
            # Reset gradients for next step
            self.optimizer.zero_grad()
            
            # Print
            if it % 50 == 0:
                elapsed = timeit.default_timer() - start_time
                self.loss_log.append(loss.cpu().data.numpy())
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss.cpu().data.numpy(), elapsed))
                start_time = timeit.default_timer()
        
    # Evaluates predictions at test points    
    def predict(self, X_star, Nt):
        X_star = torch.from_numpy(X_star).type(self.dtype)
        y_star = self.forward_pass(X_star, Nt)
        y_star = y_star.cpu().data.numpy()
        return y_star
        
if __name__ == "__main__":     
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.size': 16,
                         'lines.linewidth': 2,
                         'axes.labelsize': 20,
                         'axes.titlesize': 20,
                         'xtick.labelsize': 16,
                         'ytick.labelsize': 16,
                         'legend.fontsize': 20,
                         'axes.linewidth': 2,
                        "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
                        "text.usetex": True,                # use LaTeX to write all text
                         })
    dpiv = 100
    
    Nsim = 100 # Number of models to be trained
    Nsave = 0 # The first Nsave models will be discarded and not saved
    
    gamma = 1/6.5
    delta = 1/2.5
    
    Nt = 81 # Number of days of the infections data
    
    Ntrain = 66 # Number of days of the infections data to be used for training
    
    time = np.arange(0,Nt, 1)
    
    times = time/(Nt-1)
    dt = times[1]-times[0]
    t_train = times[:Ntrain]
    t_train = np.reshape(t_train, (-1, 1))
    
    Nm = 9 # Number of mobility and social behavior parameters
        
    Nc_all = 203 # Number of counties considered in the available data
    Nc = 203 # Number of counties to consider from all available ones
    batch_county = 60 # County batch size
    
    Ymin = 0 # Minimum value of the cumulative cases on the last day of the available data that needs to be satisfied by each county to be included in errors computation
          
    # Load processed data
    X_train = np.load('processed/X_train.npy')
    Y_train = np.load('processed/Y_train.npy')           
    
    # Randomly select Nc counties from the Nc_all available ones
    inn = np.random.permutation(np.arange(Nc_all))
    X_train = X_train[inn[:Nc],:,:]
    Y_train = Y_train[inn[:Nc],:]
    
    Nerr = sum(Y_train[:,-1]>Ymin)
    print('Number of big counties: ', Nerr)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    # Definition of parameters
    hidden_dim = 120 # LSTM hidden state dimension
    Niter = 20000 # Number of iterations for models training
    
    # Matrices needed for the trapezoidal rule based loss
    expli = torch.diag(-torch.ones(t_train.shape[0] - 2), diagonal = -1) + torch.eye(t_train.shape[0]-1)
    trap = torch.diag(torch.ones(t_train.shape[0] - 2), diagonal = -1)/2 + torch.eye(t_train.shape[0]-1)/2
        
    batch_size = t_train.shape[0] - 1
    tau = 21 # the time lag in days to consider between the mobility and social behavior parameters and the number of cumulative cases
    is_pred = 1 # 1 if loss is defined based on predictions of I(t), 0 if loss is based on the residual of I(t)
   
    loss_list = []
    
    I0 = Y_train[:,0]
    I0 = I0[:,None,None]
    R0 = np.zeros((Nc,1,1))   
        
    model = NeuralNet(X_train[:,0:tau+Ntrain-1,:], Y_train[:,:Ntrain], I0, hidden_dim, expli, trap, is_pred, batch_size, batch_county, Nm, Nc, Ntrain, tau, dt, Nt)
    
    errap = []
    errae_n = []
    
    for s in range(Nsim):

        print('Sim :', s)
        I0 = Y_train[:,0]
        I0 = I0[:,None,None]
        R0 = np.zeros((Nc,1,1))   
        
        if s>0:
            # Initialize new model parameters 
            model.U_f, model.b_f, model.W_f, \
            model.U_i, model.b_i, model.W_i, \
            model.U_s, model.b_s, model.W_s, \
            model.U_o, model.b_o, model.W_o, \
            model.V, model.c, model.E0 = model.initialize_LSTM()
            # Re-define optimizer
            model.optimizer = torch.optim.Adam([model.U_f, model.b_f,model. W_f, 
                                                model.U_i, model.b_i, model.W_i, 
                                                model.U_s, model.b_s, model.W_s, 
                                                model.U_o, model.b_o, model.W_o, 
                                                model.V, model.c, model.E0], lr=1e-3, weight_decay=0)
            model.loss_log = []
            
        if s<Nsave:
            model.train(nIter = 1)
        else:
            model.train(nIter = Niter)
            
            # Save trained model parameters and loss
            def get_params():
                
                Weights = torch.reshape( model.U_f, (Nm*hidden_dim, 1))
                Weights = torch.cat( ( Weights , torch.reshape( model.U_i, (Nm*hidden_dim, 1) ) ) , 0 )
                Weights = torch.cat( ( Weights , torch.reshape( model.U_s, (Nm*hidden_dim, 1) ) ) , 0 )
                Weights = torch.cat( ( Weights , torch.reshape( model.U_o, (Nm*hidden_dim, 1) ) ) , 0 )
            
                Weights = torch.cat( ( Weights , torch.transpose( model.b_f , 0, 1 ) ) , 0 )
                Weights = torch.cat( ( Weights , torch.transpose( model.b_i , 0, 1 ) ) , 0 )
                Weights = torch.cat( ( Weights , torch.transpose( model.b_s , 0, 1 ) ) , 0 )
                Weights = torch.cat( ( Weights , torch.transpose( model.b_o , 0, 1 ) ) , 0 )
                
                Weights = torch.cat( ( Weights , torch.reshape( model.W_f, (hidden_dim**2, 1) ) ) , 0 )
                Weights = torch.cat( ( Weights , torch.reshape( model.W_i, (hidden_dim**2, 1) ) ) , 0 )
                Weights = torch.cat( ( Weights , torch.reshape( model.W_s, (hidden_dim**2, 1) ) ) , 0 )
                Weights = torch.cat( ( Weights , torch.reshape( model.W_o, (hidden_dim**2, 1) ) ) , 0 )
            
                Weights = torch.cat( ( Weights , model.V ) , 0 )
                Weights = torch.cat( ( Weights , model.c ) , 0 )
            
                Weights = torch.cat( ( Weights , torch.reshape( model.E0, (Nc, 1) ) ) , 0 )
                
                return Variable( Weights , requires_grad=False)
        
            ww = get_params()
            ww = ww.cpu().data.numpy()
            yy = 'weights/weights_' + str(s)
            np.save(yy, ww)
            
            loss_list.append( np.log10(np.array(model.loss_log)) )
            
            # Compute prediction and extrapolation errors
            E0 = 3*I0*sigmoid(model.E0.cpu().data.numpy())
            
            def matrix_diag(diagonal):
                N = diagonal.shape[-1]
                shape = diagonal.shape[:-1] + (N, N)
                device, dtype = diagonal.device, diagonal.dtype
                result = torch.zeros(shape, dtype=dtype, device=device)
                indices = torch.arange(result.numel(), device=device).reshape(shape)
                indices = indices.diagonal(dim1=-2, dim2=-1)
                result.view(-1)[indices] = diagonal
                return result
            
            if torch.cuda.is_available() == True:
                dtype = torch.cuda.FloatTensor
            else:
                dtype = torch.FloatTensor
            E0 = torch.from_numpy(E0).type(dtype)
            I0 = torch.from_numpy(I0).type(dtype)
            R0 = torch.from_numpy(R0).type(dtype)
            S0 = 1-E0-I0-R0
            
            expli = expli.type(dtype)
            trap = trap.type(dtype)
                    
            X = X_train
            Y = torch.from_numpy(Y_train).type(dtype)
            batch_county = Nc
            
            p0 = gamma
            p1 = delta
            
            # Compute predictions over training time interval with trained model
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
            
            # Compute extrapolations over future time instances with trained model
            def Newton_system(Feval, Jac, x, eps,xprev,beta,betaprev,dt,gamma,delta):
                F_value = Feval(x,xprev,beta,betaprev,dt,gamma,delta)
                F_norm = np.linalg.norm(F_value)
                
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
            
            # Compute predictin and extrapolation errors
            for i in range(Nc):
                if Y_train[i,-1]>Ymin:
                    yinit = np.array([Sm.cpu().data.numpy()[i,-1,0], Em.cpu().data.numpy()[i,-1,0], pred[i,-1,0], Rm[i,-1,0]]) # np.array([1.0, 1.0])
                
                    beta_pred = np.squeeze(model.predict(X_train[i:i+1,Ntrain-1:,:],Nt-Ntrain+1))
                    epN = 1e-3
                    Ypred = myTrap(yinit,time[Ntrain-1:],beta_pred,1,gamma,delta,epN)
                    
                    Ipred = np.concatenate((pred[i,:,0],Ypred[1:,2]),axis=0)
                    Rpred = np.concatenate((Rm[i,:,0],Ypred[1:,3]),axis=0)
                    
                    errl = np.mean ( np.abs(Ipred[:Ntrain]+Rpred[:Ntrain]-Y_train[i,:Ntrain].T)/np.max(Y_train[i,:Ntrain]) )
                    errap.append(errl)
                    
                    if Nt>Ntrain:
                        errl = np.mean ( np.abs(Ipred[Ntrain:]+Rpred[Ntrain:]-Y_train[i,Ntrain:].T)/np.mean(Y_train[:,Ntrain:]) )
                        errae_n.append(errl)
                    
    print('Avg pred error',np.mean( np.asarray(errap) ))
    
    if Nt>Ntrain:
        print('Avg extrap error',np.mean( np.asarray(errae_n) ))
    
    # Save plot of the loss as a function of the iteration number for the trained models
    plt.figure(figsize=(12.5,10))
    for s in range(Nsim-Nsave):  
        col = np.random.rand(3,)
        plt.plot(loss_list[s], c=col)
    plt.ylabel('$\log(\mathcal{L})$')
    plt.xlabel('Iteration / 50')
    yy = 'plots/loss.png'
    plt.savefig(yy, dpi = dpiv)
