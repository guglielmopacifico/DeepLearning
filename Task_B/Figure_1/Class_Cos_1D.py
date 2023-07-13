# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

from Common import NeuralNet

# Access the gpu (also apple MPS) if available
device = "mps" if getattr(torch,'has_mps',False) \
    else "cuda" if torch.cuda.is_available() else "cpu"

if device == "mps": device = "cpu"

DEVICE = torch.device(device)

print("Running on ", device)

# Set seed for reproducibility
np.random.seed(0)
torch.manual_seed(123)

# Define the class for the PINN
class Cos_1D(nn.Module):
    """
    In this class we define the 1D problem to reproduce the plots of the section 5.2.1 and 5.2.2 of the paper.

    The goal will be to solve the following problem:

        du/dx = cos(w * x)
        u(0) = 0

    The exact solution is:

        u(x) = 1/w * sin(w * x)
    """
    
    def __init__(self, domain_extrema, w_, n_hidden_layers_, neurons_, activation_function_):
        super(Cos_1D, self).__init__()

        # Define the domain extrema
        self.domain_extrema = domain_extrema

        # Define the frequencie
        self.w = w_

        # Define the neurons for each hidden layer
        if type(neurons_) == list:
            assert len(neurons_) == n_hidden_layers_, "Number of hidden layers do not match the number of neurons"
            self.neurons = neurons_                                         # if neurons_ is a list, then it is the number of neurons per hidden layer
        else:
            self.neurons = [neurons_ for _ in range(n_hidden_layers_)]      # if neurons_ is an integer, then it is the number of neurons per hidden layer

        # Define the number of hidden layers
        assert n_hidden_layers_ > 0, "Number of hidden layers must be greater than 0"
        self.n_hidden_layers = n_hidden_layers_

        # Define the activation function
        self.activation_function = activation_function_

        # Define the NN
        self.input_layer = nn.Linear(1, self.neurons[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons[i], self.neurons[i+1]) for i in range(self.n_hidden_layers-1)])
        self.output_layer = nn.Linear(self.neurons[-1], 1)

        # Initialize the weights
        self.xavier()

        # Number of total parameters
        self.size = np.sum([np.prod([i for i in p.shape]) for p in self.parameters()])

        # Save the model to the device
        self.to(DEVICE)
    
    def forward(self, x):
        x = self.activation_function(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation_function(layer(x))
        x = self.output_layer(x)
        return x
    
    ################################################################################################
    
    def xavier(self):
        """
        This method initializes the weights of the NN using the Xavier initialization
        """
        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                # torch.nn.init.xavier_normal_(m.weight, gain=g)
                m.bias.data.fill_(0)
        
        self.apply(init_weights)
    
    def normalize_input(self, x):
        """
        This method normalizes the input x to the range [-1, 1]
        """
        return 2*(x - torch.min(x))/(torch.max(x) - torch.min(x)) - 1
    
    def restore_output(self, u):
        """
        This method restore the output of the NN to what it should be if it was given the original input
        """
        max_value = self.domain_extrema[1]
        min_value = self.domain_extrema[0]

        return u * (max_value - min_value)/2 + (max_value+min_value)/2
    
    def unnormalize_output(self, u):
        """
        This method unnormalizes the output of the NN as explained in the paper:

            unnormalize( u(x) ) = u(x) * 1/w
        """
        return u*1/self.w
    
    def exact_solution(self, x):
        """
        This method computes the exact solution at x for any given number of multi-scales
        The solution is a sum of sines with different frequencies:

                u(x) = 1/w * sin(w * x)
        """
        return 1/self.w * torch.sin(self.w * x)
    
    ################################################################################################

    def loss_function(self, x, verbose=False):
        """ 
        This method computes the loss function for the PINN
        
        The loss in calculated using an ansatz s.t. the problems becomes unconstrained.
        This is done following the Theory of Functional Connections (TFC) approach.
        The ansatz we will use is the following:

            u(x) = tanh(w*x) * NN(x)
        """
        x_normalized = self.normalize_input(x)
        x_normalized.requires_grad = True
        # Ansatz
        u_TFC = torch.tanh(self.w * x_normalized) *  self.unnormalize_output( self.forward( x_normalized ) )

        # compute the gradient of the ansatz
        u_TFC_x = torch.autograd.grad(u_TFC, x_normalized, grad_outputs=torch.ones_like(u_TFC), create_graph=True)[0]

        loss =  (u_TFC_x - torch.cos(self.w * x)).square().mean()

        if verbose: print("Loss: ", loss.item())

        return loss
    
    def loss_function_Vanilla(self, x, verbose=False):
        """ 
        This method computes the loss function for the PINN
        
        This is the classic PINN loss function, given by:
                loss = loss_ODE + loss_IC
        """
        # compute the NN output
        u = self.forward(x)

        # compute the gradient of the ansatz
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        loss_ODE =  (u_x - torch.cos(self.w * x)).square().mean()

        x_0 = torch.tensor([0], dtype=torch.float32, device=DEVICE, requires_grad=True).reshape(-1, 1)
        loss_IC = (self(x_0) - 0).square().mean()

        if verbose: print("Loss_ODE: ", loss_ODE.item(), " Loss_IC: ", loss_IC.item())

        return loss_ODE + loss_IC
    
    def fit(self, num_points, optimizer, num_epochs=1, verbose=False):
        """
        This methods trains the PINN using the given optimizer on the given data x
        """

        # Start timer for training
        start_time = time.time()

        # Devide the domain in num_points on which to train the NN
        x = torch.linspace(self.domain_extrema[0], self.domain_extrema[1], num_points, dtype=torch.float32, device=DEVICE, requires_grad=False).reshape(-1, 1)   # the input has to be of shape (n, 1)

        # List to save the loss
        history = []
        print_every = 100

        for epoch in range(num_epochs):
            # Start timer for epoch
            start_epoch_time = time.time()

            self.train()

            def closure():
                optimizer.zero_grad()
                loss = self.loss_function(x, verbose=verbose)
                loss.backward()

                history.append(loss.item())

                return loss
            
            optimizer.step(closure)

            # End timer for epoch
            end_epoch_time = time.time()

            if verbose and epoch % print_every == 0: print("Epoch : ", epoch, "\t Loss: ", history[-1], "\t Epoch_time: ", round(end_epoch_time - start_epoch_time), ' s')
        
        # End timer for training
        end_time = time.time()

        print("Final loss: ", history[-1], "\t Training_time: ", round(end_time - start_time)//60, ' min ', round(end_time - start_time)%60, ' s')

        return history