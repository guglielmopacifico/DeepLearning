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

class FBPINN(nn.Module):
    """
    This class implements the Finite Basis PINN (FBPINN), to solve the problem that the vanilla PINN is not able to solve.
    
    Here we will devide the domain in n subdomains of shape hyperrectangle, and in each subdomain we will defice a different NN.
    The full solution will be the sum of all these sub-solutions.
    """

    def __init__(self, domain_extrema, n_subdomains, overlap, sigma, n_hidden_layers, neurons, activation_function, w):
        super(FBPINN, self).__init__()

        # The extrema of the domain
        self.domain_extrema = domain_extrema

        # The number of subdomains
        self.n_subdomains = n_subdomains

        # The overlap between two consecutive subdomains
        self.overlap = overlap

        # The parameter defined s.t. the window function is 0 outside the overlap
        self.sigma = sigma

        # The width of each subdomain
        self.width = (self.domain_extrema[1] - self.domain_extrema[0])/self.n_subdomains

        # The number of hidden layers
        assert n_hidden_layers > 0, "Number of hidden layers must be greater than 0"
        self.n_hidden_layers = n_hidden_layers

        # The neurons for each hidden layer
        self.neurons = neurons

        # The activation function
        self.activation_function = activation_function

        # The frequency of the problem
        self.w = w

        # Do the domain decomposition
        self.make_subdomains()

        # Create the sub_NNs for each subdomain
        self.make_neural_networks()
    
    ################################################################################################

    def make_subdomains(self):
        """
        This method creates the subdomains of the domain
        And for each subdomain it creates also the list of the midpoints of the overlap
        """

        # Create the subdomains with the overlap & the midpoints of the overlap
        self.midpoints_overlap = []         # List of a&b midpoints of each overlap
        self.subdomains = []                # List of subdomains

        for i in range(self.n_subdomains):

            self.midpoints_overlap.append([self.domain_extrema[0] + i*self.width, self.domain_extrema[0] + (i+1)*self.width])

            if i != 0 and i != self.n_subdomains - 1:
                self.subdomains.append([self.domain_extrema[0] + i*self.width - self.overlap/2, self.domain_extrema[0] + (i+1)*self.width + self.overlap/2])
            elif i == 0:
                self.subdomains.append([self.domain_extrema[0] + i*self.width, self.domain_extrema[0] + (i+1)*self.width + self.overlap/2])
            else:
                self.subdomains.append([self.domain_extrema[0] + i*self.width - self.overlap/2, self.domain_extrema[0] + (i+1)*self.width])
    
    def window_function(self, x, a, b):
        """
        This method computes the window function for the given x, a, b
        Where:
            x is the input of the NN
            a is the left midpoint of the overlap
            b is the right midpoint of the overlap
        """
        # If x is a numpy array, convert it to a torch tensor
        if type(x) == np.ndarray: 
            x = torch.tensor(x, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
        
        # Compute the window function
        # If a or b are the extrema of the domain, then the window function must not be zero on that side
        if a == self.domain_extrema[0]:
            return torch.sigmoid((b - x)/self.sigma)
        elif b == self.domain_extrema[1]:
            return torch.sigmoid((x - a)/self.sigma)
        else:
            return torch.sigmoid((x - a)/self.sigma) * torch.sigmoid((b - x)/self.sigma)

    def make_neural_networks(self):
        """
        This method creates the neural network for each subdomain
        """

        # List of the NNs
        self.neural_networks = []

        for i in range(self.n_subdomains):
            self.neural_networks.append( NeuralNet(input_dimension = 1, output_dimension = 1,
                                                    n_hidden_layers = self.n_hidden_layers,
                                                    neurons = self.neurons,
                                                    regularization_param = 0.,
                                                    regularization_exp = 2.,
                                                    retrain_seed = 0
                                                    )
                                        )
    
    ################################################################################################

    def normalize_input(self, x):
        """
        This method normalizes the input x in the range [-1, 1]
        """
        return 2*(x - torch.min(x))/(torch.max(x) - torch.min(x)) - 1
    
    def unnormalize_output(self, u):
        """
        This method unnormalizes the output of the NN as explained in the paper
        multipling the output of the sub_NN, namely u(x), by 1/w

            unnormalize( u(x) ) = u(x) * 1/w
        """
        return u/self.w
    
    def exact_solution(self, x):
        """
        This method computes the exact solution of the problem
        """
        return 1/self.w * torch.sin(self.w * x)
    ################################################################################################

    def forward(self, x):
        """
        This method computes the output of the FBPINN for the given x.
        The output is computed following equation (13) in the paper:

            NN(x, theta) = sum_{i=1}^{n_subdomains} window_function(x, a_i, b_i) * unnormalization * NN_i * normalization_i(x)

        Where: * stands for the function composition

        So here the given x is the unnormalized input and this method does the normalization for each subdomain
        """

        # If x is a numpy array, convert it to a torch tensor
        if type(x) == np.ndarray: 
            x = torch.tensor(x, dtype=torch.float32, device=DEVICE).reshape(-1, 1)

        output = torch.zeros_like(x)

        for i in range(self.n_subdomains):
            window_function = self.window_function(x, self.midpoints_overlap[i][0], self.midpoints_overlap[i][1])
            output += window_function * self.unnormalize_output(self.neural_networks[i](self.normalize_input(x)))

        return output

    ################################################################################################

    def loss_function(self, x, verbose=False):
        # Ansatz
        u = torch.tanh(self.w * x) * self(x)

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]

        loss = (u_x - torch.cos(self.w * x)).square().mean()

        if verbose: print("Loss: ", loss.item())

        return loss
    
    def loss_function_Vanilla(self, x, verbose=False):
        u = self(x)

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]

        loss_ODE = (u_x - torch.cos(self.w * x)).square().mean()

        x_0 = torch.tensor([0], dtype=torch.float32, device=DEVICE, requires_grad=True)
        loss_IC = (self(x_0) - 0).square().mean()

        if verbose: print("Loss ODE: ", loss_ODE.item(), "Loss IC: ", loss_IC.item())

        return loss_ODE + loss_IC
    
    ################################################################################################

    def fit(self, num_points, num_epochs=1, verbose=False):
        """
        This method trains the FBPINN using the given optimizer on the given data x
        To train the FBPINN, we train each NN separately in its subdomain
        """
        
        # Start timer for training
        start_time = time.time()

        # Devide the domain in num_points on which to train the NN
        x = torch.linspace(self.domain_extrema[0], self.domain_extrema[1], num_points, dtype=torch.float32, device=DEVICE, requires_grad=True).reshape(-1, 1)   # the input has to be of shape (n, 1)

        # Define the optimizer
        # Make a list with the parameters of each sub_NN
        parameters = []
        for i in range(self.n_subdomains):
            parameters += self.neural_networks[i].parameters()

        optimizer = optim.Adam(parameters, lr=float(0.001))

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