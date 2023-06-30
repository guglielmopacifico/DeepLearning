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
torch.manual_seed(0)

# Define the class for the PINN

class PINN(nn.Module):
    
    def __init__(self, n_multi_scale_, w_list_, n_hidden_layers_, neurons_, activation_function_, seed_=0):
        super(PINN, self).__init__()

        # Define the frequencies
        assert n_multi_scale_ == len(w_list_), "Number of frequecies w do not match the number of multi-scale"
        self.n_multi_scale = n_multi_scale_
        self.w_list = w_list_

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

        # Define the seed
        self.seed = seed_

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
        torch.manual_seed(self.seed)

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
    
    def unnormalize_output(self, u):
        """
        This method unnormalizes the output of the NN as explained in the paper:

            unnormalize( u(x) ) = u(x) * 1/w
        """
        return u*1/self.w_list[0]
    
    def exact_solution(self, x):
        """
        This method computes the exact solution at x for any given number of multi-scales
        The solution is a sum of sines with different frequencies:

                u(x) = sum_{i=1}^{n_multi_scale} sin(w_i*x)
        """
        u_exact = 0
        for i in range(self.n_multi_scale):
            u_exact += torch.sin(self.w_list[i]*x)
        return u_exact
    
    ################################################################################################

    def loss_function(self, x):
        """ 
        This method computes the loss function for the PINN
        
        The loss in calculated using an ansatz s.t. the problems becomes unconstrained.
        This is done following the Theory of Functional Connections (TFC) approach.
        The ansatz we will use is the following:

            u(x) = tanh(w*x) * NN(x)
        """

        # normalize the input
        x = self.normalize_input(x)

        # compute the NN output
        u = self.forward(x)

        # unnormalize the output
        u = self.unnormalize_output(u)

        # compute the ansatz
        ansatz = torch.tanh(self.w_list[0]*x) * u

class Cos_1D_HardBC(nn.Module):
    """
    In this class we define the 1D problem to reproduce the plots of the section 5.2.1 and 5.2.2 of the paper.

    The goal will be to solve the following problem:

        du/dx = cos(w * x)
        u(0) = 0

    The exact solution is:

        u(x) = 1/w * sin(w * x)
    
    Furthermore, we will impose the hard boundary condition u(0) = 0 by using the ansatz:

        u(x) = tanh(w*x) * NN(x)
    """
    
    def __init__(self, domain_extrema, w_, n_hidden_layers_, neurons_, activation_function_, seed_=0):
        super(Cos_1D_HardBC, self).__init__()

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

        # Define the seed
        self.seed = seed_

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
        torch.manual_seed(self.seed)

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

        # compute the NN output
        u = self.forward(x_normalized)

        # unnormalize the output
        u = self.unnormalize_output(u)

        # compute the gradient
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0] # Here lied the error

        # compute the ansatz
        u_HardBC = torch.tanh(self.w * x) * u_x

        loss =  (u_HardBC - torch.cos(self.w * x)).square().mean()

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
    
    ################################################################################################

class FBPINN(nn.Module):
    """
    This class implements the Finite Basis PINN (FBPINN), to solve the problem that the vanilla PINN is not able to solve.
    
    Here we will devide the domain in n subdomains of shape hyperrectangle, and in each subdomain we will defice a different NN.
    The full solution will be the sum of all these sub-solutions.
    """

    def __init__(self, domain_extrema, n_subdomains, overlap, sigma, n_hidden_layers, neurons, activation_function, PINN_class, w):
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

        # The class of the PINN
        self.PINN_class = PINN_class

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
            # self.neural_networks.append(self.PINN_class(domain_extrema= self.subdomains[i], w_=self.w, n_hidden_layers_=self.n_hidden_layers, neurons_=self.neurons, activation_function_=self.activation_function))
    
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
    def compute_sub_NN_outputs(self, x, i):
        """
        This method computes the output of the sub_NN and its gradient for the given x and subdomain i
        """
        # normalize the input
        x_normalized = self.normalize_input(x)
        x_normalized.requires_grad = True

        # compute the output of the sub_NN
        u = self.neural_networks[i](x_normalized)

        # unnormalize the output
        u = self.unnormalize_output(u)

        # apply the window function
        u = self.window_function(x, self.midpoints_overlap[i][0], self.midpoints_overlap[i][1]) * u

        # compute the gradient
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(x_normalized), create_graph=True)[0]

        return u, u_x

    def loss_function(self, x, verbose=False):
        
        # compute the output of the sub_NN and its gradient for each subdomain
        u, u_x = torch.zeros_like(x), torch.zeros_like(x)
        for i in range(self.n_subdomains):
            u_i, u_x_i = self.compute_sub_NN_outputs(x, i)
            u += u_i
            u_x += u_x_i
        
        # Apply the boundary conditions wrting the ansatz
        u_TFC = torch.tanh(self.w * x) * u_x

        # Compute the loss
        loss = (u_TFC - torch.cos(self.w * x)).square().mean()

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

        x = torch.linspace(self.domain_extrema[i][0], self.domain_extrema[i][1], num_points, dtype=torch.float32, device=DEVICE, requires_grad=False).reshape(-1, 1)   # the input has to be of shape (n, 1)

        # List to save the loss
        history = []
        print_every = 100

        # Define the optimizer
        # Make a list with the parameters of each sub_NN
        parameters = []
        for i in range(self.n_subdomains):
            parameters += self.neural_networks[i].parameters()

        optimizer = optim.Adam(parameters, lr=float(0.001))

        for epoch in range(num_epochs):
            # Start timer for epoch
            start_epoch_time = time.time()

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
    
    ################################################################################################

    # def fit(self, num_points, num_epochs=1, verbose=False):
    #     """
    #     This method trains the FBPINN using the given optimizer on the given data x
    #     To train the FBPINN, we train each NN separately in its subdomain
    #     """
    #     # Devide the domain in num_points on which to train the FBPINN
    #     x = torch.linspace(self.domain_extrema[0], self.domain_extrema[1], num_points, dtype=torch.float32,     # the input has to be of shape (n, 1)
    #                        device=DEVICE, requires_grad=False).reshape(-1, 1)                                   # do not compute the gradient of the full x

    #     self.hisory_sub_NN = []    # List of the history of the sub_NN

    #     for i in range(self.n_subdomains):

    #         # Take only the points in the subdomain
    #         mask = (x >= self.subdomains[i][0]) & (x <= self.subdomains[i][1])
    #         x_subdomain = x[mask]
    #         x_subdomain.requires_grad = True        # compute the gradient of the subdomain

    #         history = []
    #         print_every = 100

    #         for epoch in range(num_epochs):
    #             # Start timer for epoch
    #             start_epoch_time = time.time()
                


    #         # Create the optimizer for the sub_NN
    #         optimizer_ADAM = optim.Adam(self.neural_networks[i].parameters(), lr=float(0.001))

    #         # Train the sub_NN and save the history
    #         self.hisory_sub_NN.append(self.neural_networks[i].fit(num_points, optimizer_ADAM, num_epochs, verbose))