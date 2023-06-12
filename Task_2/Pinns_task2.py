import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from Common import NeuralNet
import time

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(128)

class Pinns:
    """
    Class to define the Physics Informed Neural Network (PINN) for the task 2 of the project.
    
    With this class we are going to train the PINN on the entire domain [0, 8]x[0, 1],
    treating all together the 8 phases of the 2 cycles.
    """
    def __init__(self, n_int_, n_sb_, n_tb_, lambda_u_=10, coefficient_neurons_=20, coefficient_layers_=4, alpha_f_=0.005, h_f_=5, T_hot_=4, T_0_=1, U_f_=None):
        self.n_int = n_int_     # n_int_:= number of intertior points
        self.n_sb = n_sb_       # n_sb_ := number of spatial boundary points
        self.n_tb = n_tb_       # n_tb_ := number of time boundary points

        # Set the paremeters of the equation
        self.alpha_f = alpha_f_
        self.h_f = h_f_
        self.T_hot = T_hot_
        self.T_0 = T_0_
        self.U_f = U_f_

        # Extrema of the solution domain (t,x) in [0, t]x[0, L]
        self.domain_extrema = torch.tensor([[0, 8],  # Time dimension
                                            [0, 1]])  # Space dimension

        # Number of space dimensions
        self.space_dimensions = 1

        # Parameter to balance role of data and PDE
        self.lambda_u = lambda_u_

        # Number of neurons in the coefficient NN
        self.coefficient_neurons =coefficient_neurons_
        self.coefficient_layers = coefficient_layers_

        # FF Dense NN to approximate the solution of the underlying reaction-convection-diffusion equations of the fluid
        self.approximate_solution = NeuralNet(input_dimension=self.domain_extrema.shape[0], output_dimension=1, # is a NN with input_dim=2 (time & space), output_dim=1 (fluid_temp)
                                              n_hidden_layers=4,
                                              neurons=20,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=42)
        
        # FF Dense NN to approximate the solid temperature we wish to infer
        self.approximate_coefficient = NeuralNet(input_dimension=self.domain_extrema.shape[0], output_dimension=1,  # is a NN with input_dim=2 (time & space), output_dim=1 (solid_temp)
                                                 n_hidden_layers=coefficient_layers_,
                                                 neurons=coefficient_neurons_,
                                                 regularization_param=0.,
                                                 regularization_exp=2.,
                                                 retrain_seed=42)

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])   # it will create a 2 cloumns tensor, the rows nunmber is specified after every time it is used

        # Training sets S_sb, S_tb, S_int as torch dataloader
        self.training_set_sb, self.training_set_tb, self.training_set_int, self.training_set_meas = self.assemble_datasets()

    ################################################################################################
    # Function to linearly transform a tensor whose value are between 0 and 1
    # to a tensor whose values are between the domain extrema
    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]
    
    # Function Uf(t) -> given the time it gives back the velocity of the fluid in the relative phase
    def fluid_velocity(self, inputs):
        Uf = torch.full(inputs.shape, 999)  # give all 999 for semplicity when testing

        for i, t in enumerate(inputs):
            # Charging Phase
            if (t <= 1) or (t > 4 and t <=5 ): Uf[i] = 1
            # Discharging Phase
            elif (t > 2 and t <= 3) or (t > 6 and t <= 7): Uf[i] = -1
            # Idle Phase
            elif (t > 1 and t <= 2) or (t > 3 and t <= 4) or (t > 5 and t <= 6) or (t > 7 and t <= 8): Uf[i] = 0

        return Uf
    
    ################################################################################################
    # Function returning the input-output tensor required to assemble the training set S_tb corresponding to the temporal boundary
    def add_temporal_boundary_points(self):
        t0 = self.domain_extrema[0, 0]
        input_tb = self.soboleng.draw(self.n_tb)    # input_sb has two columns (t, x) both with random numbers in the two respective domains
        input_tb[:, 0] = torch.full(input_tb[:, 0].shape, t0)   # overwrite the entier column of time with t0
        output_tb = torch.full(input_tb[:, 0].shape, self.T_0).reshape(-1, 1)   # the output has 1 column

        # """QUI"""
        # print('ADD TEMPORAL BOUDARY POINTS:')
        # print('input_tb: ', input_tb.shape)
        # print('output_tb: ', output_tb.shape)

        return input_tb, output_tb  # input_tb is the sequence of x_n; output_tb is the sequence u0(x_n)

    # Function returning the input-output tensor required to assemble the training set S_sb corresponding to the spatial boundary
    def add_spatial_boundary_points(self):
        x0 = self.domain_extrema[1, 0]
        xL = self.domain_extrema[1, 1]

        # Dataset with random [t, x] both in [0, 1]
        input_sb = self.soboleng.draw(self.n_sb)

        # Assigne the spacial boundary x=x0
        input_sb_0 = torch.clone(input_sb)
        input_sb_0[:, 1] = torch.full(input_sb_0[:, 1].shape, x0)

        # Assigne the spacial boundary x=xL
        input_sb_L = torch.clone(input_sb)
        input_sb_L[:, 1] = torch.full(input_sb_L[:, 1].shape, xL)

        # Def a tensor to add the delta in time to each of the input dataset and have the different phases
        # This tenosr is full of [0, 0] and it will be filled on the t-column with the relative delta_t
        delta_time = torch.zeros_like(input_sb)

        # We are going now to define the input dataset for all the different phases over the 2 cycles.
        # However, not that even if the input for the same phase over the 2 cycles is different (the t)
        # the output is always the same (the spatial boundary conditions are the same), so we just need one output per phase.
        """Charging Phase"""
        # First charging phase -> t in [0, 1] => delta_t=0
        input_sb_0_charging_1 = torch.clone(input_sb_0)
        input_sb_L_charging_1 = torch.clone(input_sb_L)
        
        # Second charging phase -> t in [4, 5] => delta_t=4
        delta_time[:, 0] = torch.full(delta_time[:, 0].shape, 4)

        input_sb_0_charging_2 = torch.clone(input_sb_0)
        input_sb_0_charging_2 = input_sb_0_charging_2 + delta_time

        input_sb_L_charging_2 = torch.clone(input_sb_L)
        input_sb_L_charging_2 = input_sb_L_charging_2 + delta_time

        # Output charging phase
        output_sb_0_charging = torch.full(input_sb_0[:, 0].shape, self.T_hot).reshape(-1, 1)
        output_sb_L_charging = torch.full(input_sb_L[:, 0].shape, 0).reshape(-1, 1)

        """Discharging Phase"""
        # First discharging phase -> t in [2, 3] => delta_t=2
        delta_time[:, 0] = torch.full(delta_time[:, 0].shape, 2)

        input_sb_0_discharging_1 = torch.clone(input_sb_0)
        input_sb_0_discharging_1 = input_sb_0_discharging_1 + delta_time

        input_sb_L_discharging_1 = torch.clone(input_sb_L)
        input_sb_L_discharging_1 = input_sb_L_discharging_1 + delta_time

        # Second discharging phase -> t in [6, 7] => delta_t=6
        delta_time[:, 0] = torch.full(delta_time[:, 0].shape, 6)

        input_sb_0_discharging_2 = torch.clone(input_sb_0)
        input_sb_0_discharging_2 = input_sb_0_discharging_2 + delta_time

        input_sb_L_discharging_2 = torch.clone(input_sb_L)
        input_sb_L_discharging_2 = input_sb_L_discharging_2 + delta_time

        # Output discharging phase
        output_sb_0_discharging = torch.full(input_sb_0[:, 0].shape, 0).reshape(-1, 1)
        output_sb_L_discharging = torch.full(input_sb_L[:, 0].shape, self.T_0).reshape(-1, 1)


        """Idle Phase"""
        # First idle phase -> t in [1, 2] => delta_t=1
        delta_time[:, 0] = torch.full(delta_time[:, 0].shape, 1)

        input_sb_0_idle_1 = torch.clone(input_sb_0)
        input_sb_0_idle_1 = input_sb_0_idle_1 + delta_time

        input_sb_L_idle_1 = torch.clone(input_sb_L)
        input_sb_L_idle_1 = input_sb_L_idle_1 + delta_time

        # Second idle phase -> t in [3, 4] => delta_t=3
        delta_time[:, 0] = torch.full(delta_time[:, 0].shape, 3)

        input_sb_0_idle_2 = torch.clone(input_sb_0)
        input_sb_0_idle_2 = input_sb_0_idle_2 + delta_time

        input_sb_L_idle_2 = torch.clone(input_sb_L)
        input_sb_L_idle_2 = input_sb_L_idle_2 + delta_time

        # Third idle phase -> t in [5, 6] => delta_t=5
        delta_time[:, 0] = torch.full(delta_time[:, 0].shape, 5)

        input_sb_0_idle_3 = torch.clone(input_sb_0)
        input_sb_0_idle_3 = input_sb_0_idle_3 + delta_time

        input_sb_L_idle_3 = torch.clone(input_sb_L)
        input_sb_L_idle_3 = input_sb_L_idle_3 + delta_time

        # Fourth idle phase -> t in [7, 8] => delta_t=7
        delta_time[:, 0] = torch.full(delta_time[:, 0].shape, 7)

        input_sb_0_idle_4 = torch.clone(input_sb_0)
        input_sb_0_idle_4 = input_sb_0_idle_4 + delta_time

        input_sb_L_idle_4 = torch.clone(input_sb_L)
        input_sb_L_idle_4 = input_sb_L_idle_4 + delta_time

        # Output idle phase
        output_sb_0_idle = torch.full(input_sb_0[:, 0].shape, 0).reshape(-1, 1)
        output_sb_L_idle = torch.full(input_sb_L[:, 0].shape, 0).reshape(-1, 1)
        
        # requires the grad for input tensors as we will have to compute the derivatives
        return torch.cat([  # cycle 1
                        input_sb_0_charging_1.requires_grad_(True), input_sb_L_charging_1.requires_grad_(True),
                        input_sb_0_idle_1.requires_grad_(True), input_sb_L_idle_1.requires_grad_(True),
                        input_sb_0_discharging_1.requires_grad_(True), input_sb_L_discharging_1.requires_grad_(True),
                        input_sb_0_idle_2.requires_grad_(True), input_sb_L_idle_2.requires_grad_(True),
                            # cycle 2
                        input_sb_0_charging_2.requires_grad_(True), input_sb_L_charging_2.requires_grad_(True),
                        input_sb_0_idle_3.requires_grad_(True), input_sb_L_idle_3.requires_grad_(True),
                        input_sb_0_discharging_2.requires_grad_(True), input_sb_L_discharging_2.requires_grad_(True),
                        input_sb_0_idle_4.requires_grad_(True), input_sb_L_idle_4.requires_grad_(True)
                        ], 0), torch.cat([  # cycle 1
                                        output_sb_0_charging, output_sb_L_charging,
                                        output_sb_0_idle, output_sb_L_idle,
                                        output_sb_0_discharging, output_sb_L_discharging,
                                        output_sb_0_idle, output_sb_L_idle,
                                            # cycle 2
                                        output_sb_0_charging, output_sb_L_charging,
                                        output_sb_0_idle, output_sb_L_idle,
                                        output_sb_0_discharging, output_sb_L_discharging,
                                        output_sb_0_idle, output_sb_L_idle
                                        ], 0)

    # Function returning the input-output tensor required to assemble the training set S_int corresponding to the interior domain where the PDE is enforced
    def add_interior_points(self):
        # Now we use the convert fct since we want the t is in [0, 8]
        input_int = self.convert(self.soboleng.draw(self.n_int))
        output_int = torch.zeros((input_int.shape[0], 1))
        
        return input_int, output_int
    
    # Function returning the input-output tensor required to assemble the training set S_meas corresponding to the measured points in the domain.
    # These points are read from the file "DataSolution.txt"
    def get_measurement_data(self):
        # Read the CSV file into a DataFrame
        df_meas = pd.read_csv('DataSolution.txt')

        # Convert the DataFrame to a torch.tensor
        tensor_meas = torch.tensor(df_meas.values , dtype=torch.float)
        
        # The first 2 columns are the inputs: [t, x]
        input_meas = tensor_meas[:, :2]
        
        # The last column are the outputs: [Tf]
        output_meas = tensor_meas[:, 2:]
        return input_meas, output_meas

    # Function returning the training sets S_sb, S_tb, S_int as dataloader
    def assemble_datasets(self):
        input_sb, output_sb = self.add_spatial_boundary_points()    # S_sb
        input_tb, output_tb = self.add_temporal_boundary_points()   # S_tb
        input_int, output_int = self.add_interior_points()          # S_int
        input_meas, output_meas = self.get_measurement_data()       # S_meas

        training_set_sb = DataLoader(torch.utils.data.TensorDataset(input_sb, output_sb), batch_size=16*self.space_dimensions*self.n_sb, shuffle=False)  #batch_size has *8 since there are 8 different phases and for each one we have 2 conditions (x0 & xL)
        training_set_tb = DataLoader(torch.utils.data.TensorDataset(input_tb, output_tb), batch_size=self.n_tb, shuffle=False)
        training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int, output_int), batch_size=self.n_int, shuffle=False)
        training_set_meas = DataLoader(torch.utils.data.TensorDataset(input_meas, output_meas), batch_size=output_meas.shape[0], shuffle=False)

        return training_set_sb, training_set_tb, training_set_int, training_set_meas

    ################################################################################################
    # Function to compute the terms required in the definition of the TEMPORAL boundary residual
    def apply_initial_condition(self, input_tb):
        u_pred_tb = self.approximate_solution(input_tb)
        
        return u_pred_tb

    # Function to compute the terms required in the definition of the SPATIAL boundary residual
    def apply_boundary_conditions(self, input_sb):
        # input_tb is a tensor of size [16*self.n_sb, 2]
        # as defined in "add_spatial_boundary_points" we have 2 boundary conditions for each phase
        # we then have to devide the input_sb in 16
        assert (input_sb.requires_grad==True)   # make sure the grad is requested so we can compute the derivatives

        # Devide all the input datasets
            # cycle 1
        input_sb_0_charging_1 = input_sb[:int(input_sb.shape[0]/16), :]
        input_sb_L_charging_1 = input_sb[int(input_sb.shape[0]/16):2*int(input_sb.shape[0]/16), :]
        input_sb_0_idle_1 = input_sb[2*int(input_sb.shape[0]/16):3*int(input_sb.shape[0]/16), :]
        input_sb_L_idle_1 = input_sb[3*int(input_sb.shape[0]/16):4*int(input_sb.shape[0]/16), :]
        input_sb_0_discharging_1 = input_sb[4*int(input_sb.shape[0]/16):5*int(input_sb.shape[0]/16), :]
        input_sb_L_discharging_1 = input_sb[5*int(input_sb.shape[0]/16):6*int(input_sb.shape[0]/16), :]
        input_sb_0_idle_2 = input_sb[6*int(input_sb.shape[0]/16):7*int(input_sb.shape[0]/16), :]
        input_sb_L_idle_2 = input_sb[7*int(input_sb.shape[0]/16):8*int(input_sb.shape[0]/16), :]
            # cycle 2
        input_sb_0_charging_2 = input_sb[8*int(input_sb.shape[0]/16):9*int(input_sb.shape[0]/16), :]
        input_sb_L_charging_2 = input_sb[9*int(input_sb.shape[0]/16):10*int(input_sb.shape[0]/16), :]
        input_sb_0_idle_3 = input_sb[10*int(input_sb.shape[0]/16):11*int(input_sb.shape[0]/16), :]
        input_sb_L_idle_3 = input_sb[11*int(input_sb.shape[0]/16):12*int(input_sb.shape[0]/16), :]
        input_sb_0_discharging_2 = input_sb[12*int(input_sb.shape[0]/16):13*int(input_sb.shape[0]/16), :]
        input_sb_L_discharging_2 = input_sb[13*int(input_sb.shape[0]/16):14*int(input_sb.shape[0]/16), :]
        input_sb_0_idle_4 = input_sb[14*int(input_sb.shape[0]/16):15*int(input_sb.shape[0]/16), :]
        input_sb_L_idle_4 = input_sb[15*int(input_sb.shape[0]/16):, :]

        """Charging Phase"""
        # First charging phase
            # x0 -> compute Tf
        u_pred_sb_0_charging_1 = self.approximate_solution(input_sb_0_charging_1).reshape(-1, 1)

            # xL -> compute dTf/dx
        u_pred_Tf = self.approximate_solution(input_sb_L_charging_1)
        u_pred_sb_L_charging_1 = torch.autograd.grad(u_pred_Tf.sum(), input_sb_L_charging_1, create_graph=True)[0][:, 1].reshape(-1, 1) # take only d/dx

        # Second charging phase
            # x0 -> compute Tf
        u_pred_sb_0_charging_2 = self.approximate_solution(input_sb_0_charging_2).reshape(-1, 1)

            # xL -> compute dTf/dx
        u_pred_Tf = self.approximate_solution(input_sb_L_charging_2)
        u_pred_sb_L_charging_2 = torch.autograd.grad(u_pred_Tf.sum(), input_sb_L_charging_2, create_graph=True)[0][:, 1].reshape(-1, 1) # take only d/dx

        """Discharging Phase"""
        # First discharging phase
            # x0 -> compute dTf/dx
        u_pred_Tf = self.approximate_solution(input_sb_0_discharging_1)
        u_pred_sb_0_discharging_1 = torch.autograd.grad(u_pred_Tf.sum(), input_sb_0_discharging_1, create_graph=True)[0][:, 1].reshape(-1, 1) # take only d/dx
        
            # xL -> compute Tf
        u_pred_sb_L_discharging_1 = self.approximate_solution(input_sb_L_discharging_1).reshape(-1, 1)

        # Second discharging phase
            # x0 -> compute dTf/dx
        u_pred_Tf = self.approximate_solution(input_sb_0_discharging_2)
        u_pred_sb_0_discharging_2 = torch.autograd.grad(u_pred_Tf.sum(), input_sb_0_discharging_2, create_graph=True)[0][:, 1].reshape(-1, 1) # take only d/dx
        
            # xL -> compute Tf
        u_pred_sb_L_discharging_2 = self.approximate_solution(input_sb_L_discharging_2).reshape(-1, 1)

        """Idle Phase"""
        # First idle phase
            # x0 -> compute dTf/dx
        u_pred_Tf = self.approximate_solution(input_sb_0_idle_1)
        u_pred_sb_0_idle_1 = torch.autograd.grad(u_pred_Tf.sum(), input_sb_0_idle_1, create_graph=True)[0][:, 1].reshape(-1, 1) # take only d/dx

            # xL -> compute dTf/dx
        u_pred_Tf = self.approximate_solution(input_sb_L_idle_1)
        u_pred_sb_L_idle_1 = torch.autograd.grad(u_pred_Tf.sum(), input_sb_L_idle_1, create_graph=True)[0][:, 1].reshape(-1, 1) # take only d/dx

        # Second idle phase
            # x0 -> compute dTf/dx
        u_pred_Tf = self.approximate_solution(input_sb_0_idle_2)
        u_pred_sb_0_idle_2 = torch.autograd.grad(u_pred_Tf.sum(), input_sb_0_idle_2, create_graph=True)[0][:, 1].reshape(-1, 1) # take only d/dx

            # xL -> compute dTf/dx
        u_pred_Tf = self.approximate_solution(input_sb_L_idle_2)
        u_pred_sb_L_idle_2 = torch.autograd.grad(u_pred_Tf.sum(), input_sb_L_idle_2, create_graph=True)[0][:, 1].reshape(-1, 1) # take only d/dx

        # Third idle phase
            # x0 -> compute dTf/dx
        u_pred_Tf = self.approximate_solution(input_sb_0_idle_3)
        u_pred_sb_0_idle_3 = torch.autograd.grad(u_pred_Tf.sum(), input_sb_0_idle_3, create_graph=True)[0][:, 1].reshape(-1, 1) # take only d/dx

            # xL -> compute dTf/dx
        u_pred_Tf = self.approximate_solution(input_sb_L_idle_3)
        u_pred_sb_L_idle_3 = torch.autograd.grad(u_pred_Tf.sum(), input_sb_L_idle_3, create_graph=True)[0][:, 1].reshape(-1, 1) # take only d/dx

        # Fourth idle phase
            # x0 -> compute dTf/dx
        u_pred_Tf = self.approximate_solution(input_sb_0_idle_4)
        u_pred_sb_0_idle_4 = torch.autograd.grad(u_pred_Tf.sum(), input_sb_0_idle_4, create_graph=True)[0][:, 1].reshape(-1, 1) # take only d/dx

            # xL -> compute dTf/dx
        u_pred_Tf = self.approximate_solution(input_sb_L_idle_4)
        u_pred_sb_L_idle_4 = torch.autograd.grad(u_pred_Tf.sum(), input_sb_L_idle_4, create_graph=True)[0][:, 1].reshape(-1, 1) # take only d/dx

        return torch.cat([  # cycle 1
                        u_pred_sb_0_charging_1, u_pred_sb_L_charging_1,
                        u_pred_sb_0_idle_1, u_pred_sb_L_idle_1,
                        u_pred_sb_0_discharging_1, u_pred_sb_L_discharging_1,
                        u_pred_sb_0_idle_2, u_pred_sb_L_idle_2,
                            # cycle 2
                        u_pred_sb_0_charging_2, u_pred_sb_L_charging_2,
                        u_pred_sb_0_idle_3, u_pred_sb_L_idle_3,
                        u_pred_sb_0_discharging_2, u_pred_sb_L_discharging_2,
                        u_pred_sb_0_idle_4, u_pred_sb_L_idle_4
                        ], 0)

    # Function to compute the PDE residuals
    def compute_pde_residual(self, input_int):
        input_int.requires_grad = True
        u = self.approximate_solution(input_int).reshape(-1,)       # u is the solution (Tf) of the PDE
        g = self.approximate_coefficient(input_int).reshape(-1,)    # g is the function (Ts) that is requested

        # grad compute the gradient of a "SCALAR" function L with respect to some input nxm TENSOR Z=[[x1, y1],[x2,y2],[x3,y3],...,[xn,yn]], m=2
        # it returns grad_L = [[dL/dx1, dL/dy1],[dL/dx2, dL/dy2],[dL/dx3, dL/dy3],...,[dL/dxn, dL/dyn]]
        # Note: pytorch considers a tensor [u1, u2,u3, ... ,un] a vectorial function
        # whereas sum_u = u1 + u2 + u3 + u4 + ... + un as a "scalar" one

        # In our case ui = u(xi), therefore the line below returns:
        # grad_u = [[dsum_u/dx1, dsum_u/dy1],[dsum_u/dx2, dsum_u/dy2],[dsum_u/dx3, dL/dy3],...,[dsum_u/dxm, dsum_u/dyn]]
        # and dsum_u/dxi = d(u1 + u2 + u3 + u4 + ... + un)/dxi = d(u(x1) + u(x2) u3(x3) + u4(x4) + ... + u(xn))/dxi = dui/dxi

        # Since u for us is u = (uf, us), we have to devide the two cases

        grad_u = torch.autograd.grad(u.sum(), input_int, create_graph=True)[0]
        grad_u_t = grad_u[:, 0]
        grad_u_x = grad_u[:, 1]
        grad_u_xx = torch.autograd.grad(grad_u_x.sum(), input_int, create_graph=True)[0][:, 1]

        # Compute the velocity of the fluid Uf(t)
        Uf = self.fluid_velocity(input_int[:, 0])

        residual = (grad_u_t + Uf*grad_u_x) - (self.alpha_f*grad_u_xx - self.h_f*(u-g))

        return residual.reshape(-1, )

    # Function to compute the total loss (weighted sum of spatial boundary loss, temporal boundary loss and interior loss)
    def compute_loss(self, inp_train_sb, u_train_sb, inp_train_tb, u_train_tb, inp_train_int, inp_train_meas, u_train_meas, verbose=True):
        u_pred_sb = self.apply_boundary_conditions(inp_train_sb)
        u_pred_tb = self.apply_initial_condition(inp_train_tb)
        u_pred_meas = self.approximate_solution(inp_train_meas)

        assert (u_pred_sb.shape[1] == u_train_sb.shape[1])
        assert (u_pred_tb.shape[1] == u_train_tb.shape[1])
        assert (u_pred_meas.shape[1] == u_train_meas.shape[1])

        r_int = self.compute_pde_residual(inp_train_int)
        r_sb = u_train_sb - u_pred_sb
        r_tb = u_train_tb - u_pred_tb
        r_meas = u_train_meas - u_pred_meas

        loss_sb = torch.mean(abs(r_sb) ** 2)
        loss_tb = torch.mean(abs(r_tb) ** 2)
        loss_int = torch.mean(abs(r_int) ** 2)
        loss_meas = torch.mean(abs(r_meas) ** 2)

        loss_u = loss_sb + loss_tb + loss_meas

        loss = torch.log10(self.lambda_u * loss_u + loss_int)

        if verbose: print("Total loss: ", round(loss.item(), 4), "| Boundary Loss: ", round(torch.log10(loss_int).item(), 4), "| Measure Loss: ", round(torch.log10(loss_meas).item(), 4), "| Function Loss: ", round(torch.log10(loss_u).item(), 4))

        # Save the losses in some attributes
        self.total_loss = round(loss.item(), 4)
        self.boundary_loss = round(torch.log10(loss_int).item(), 4)
        self.measure_loss = round(torch.log10(loss_meas).item(), 4)
        self.function_loss = round(torch.log10(loss_u).item(), 4)

        return loss

    ################################################################################################
    def fit(self, num_epochs, optimizer, verbose=True):
        # Start the timer
        start_time = time.time()

        history = list()

        # Loop over epochs
        for epoch in range(num_epochs):
            if verbose: print("################################ ", epoch, " ################################")

            for j, ((inp_train_sb, u_train_sb), (inp_train_tb, u_train_tb), (inp_train_int, u_train_int), (inp_train_meas, u_train_meas)) in enumerate(zip(self.training_set_sb, self.training_set_tb, self.training_set_int, self.training_set_meas)):
                def closure():
                    optimizer.zero_grad()
                    loss = self.compute_loss(inp_train_sb, u_train_sb, inp_train_tb, u_train_tb, inp_train_int, inp_train_meas, u_train_meas, verbose=verbose)
                    loss.backward()

                    history.append(loss.item())
                    return loss
                
                optimizer.step(closure=closure)

        print('Final Loss: ', history[-1])
        print("Run time for the fitting: ", round((time.time()- start_time)/60), " minutes", round((time.time()- start_time)%60), " seconds")

        # Save the run time in an attribute
        self.run_time = (time.time()- start_time)
        return history

    ################################################################################################
    def plotting(self, vmax_=None, savefig=False):
        inputs = self.soboleng.draw(100000)
        inputs = self.convert(inputs)

        output_tf = self.approximate_solution(inputs).reshape(-1, )
        output_ts = self.approximate_coefficient(inputs).reshape(-1, )

        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        if vmax_ is None: im1 = axs[0].scatter(inputs[:, 0].detach(), inputs[:, 1].detach(), c=output_tf.detach(), cmap="jet")
        else: im1 = axs[0].scatter(inputs[:, 0].detach(), inputs[:, 1].detach(), c=output_tf.detach(), cmap="jet", vmax=vmax_)
        axs[0].set_xlabel("t")
        axs[0].set_ylabel("x")
        plt.colorbar(im1, ax=axs[0])
        axs[0].grid(True, which="both", ls=":")
        if vmax_ is None: im2 = axs[1].scatter(inputs[:, 0].detach(), inputs[:, 1].detach(), c=output_ts.detach(), cmap="jet")
        else: im2 = axs[1].scatter(inputs[:, 0].detach(), inputs[:, 1].detach(), c=output_ts.detach(), cmap="jet", vmax=vmax_)
        axs[1].set_xlabel("t")
        axs[1].set_ylabel("x")
        plt.colorbar(im2, ax=axs[1])
        axs[1].grid(True, which="both", ls=":")
        axs[0].set_title(r"Approximate Solution $T_f$")
        axs[1].set_title(r"Approximate Solution $T_s$")

        fig.suptitle(f"lambda_u = {self.lambda_u} neurons = {self.coefficient_neurons}\nn_int = {self.n_int} n_sb = {self.n_sb} n_tb = {self.n_tb}", size=18)

        if savefig :plt.savefig(f"Plots/lambda_u={self.lambda_u}_neurons={self.coefficient_neurons}_n_int={self.n_int}_n_sb={self.n_sb}_n_tb={self.n_tb}.png")

        plt.show()

class Pinns_Cycle:
    """
    Class to define the Physics Informed Neural Network (PINN) for the task 2 of the project.
    
    With this class we are going to train the PINN on a specific cycle,
    the idea here to to split the different phases of the cycles and train the PINN on each of them.
    """
    def __init__(self, n_int_, n_sb_, n_tb_, t0_=0, tf_=8, lambda_u_=10, coefficient_neurons_=20, coefficient_layers_=4, alpha_f_=0.005, h_f_=5, T_hot_=4, T_0_=1, U_f_=None):
        self.n_int = n_int_     # n_int_:= number of intertior points
        self.n_sb = n_sb_       # n_sb_ := number of spatial boundary points
        self.n_tb = n_tb_       # n_tb_ := number of time boundary points

        # Set the paremeters of the equation
        self.alpha_f = alpha_f_
        self.h_f = h_f_
        self.T_hot = T_hot_
        self.T_0 = T_0_
        self.U_f = U_f_

        # Extrema of the solution domain (t,x) in [0, t]x[0, L]
        self.domain_extrema = torch.tensor([[t0_, tf_],  # Time dimension
                                            [0, 1]])  # Space dimension

        # Number of space dimensions
        self.space_dimensions = 1

        # Parameter to balance role of data and PDE
        self.lambda_u = lambda_u_

        # Number of neurons in the coefficient NN
        self.coefficient_neurons =coefficient_neurons_
        self.coefficient_layers = coefficient_layers_

        # FF Dense NN to approximate the solution of the underlying reaction-convection-diffusion equations of the fluid
        self.approximate_solution = NeuralNet(input_dimension=self.domain_extrema.shape[0], output_dimension=1, # is a NN with input_dim=2 (time & space), output_dim=1 (fluid_temp)
                                              n_hidden_layers=4,
                                              neurons=20,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=42)
        
        # FF Dense NN to approximate the solid temperature we wish to infer
        self.approximate_coefficient = NeuralNet(input_dimension=self.domain_extrema.shape[0], output_dimension=1,  # is a NN with input_dim=2 (time & space), output_dim=1 (solid_temp)
                                                 n_hidden_layers=coefficient_layers_,
                                                 neurons=coefficient_neurons_,
                                                 regularization_param=0.,
                                                 regularization_exp=2.,
                                                 retrain_seed=42)

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])   # it will create a 2 cloumns tensor, the rows nunmber is specified after every time it is used

        # Training sets S_sb, S_tb, S_int as torch dataloader
        self.training_set_sb, self.training_set_tb, self.training_set_int, self.training_set_meas = self.assemble_datasets()

    ################################################################################################
    # Function to linearly transform a tensor whose value are between 0 and 1
    # to a tensor whose values are between the domain extrema
    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]
    
    # Function Uf(t) -> given the time it gives back the velocity of the fluid in the relative phase
    def fluid_velocity(self, inputs):
        Uf = torch.full(inputs.shape, 999)  # give all 999 for semplicity when testing

        # Charging Phase
        if (self.domain_extrema[0, 1] <= 1) or (self.domain_extrema[0, 0] >= 4 and self.domain_extrema[0, 1] <=5 ): Uf = torch.full(inputs.shape, 1)
        
        # Discharging Phase
        elif (self.domain_extrema[0, 0] >= 2 and self.domain_extrema[0, 1] <= 3) or (self.domain_extrema[0, 0] >= 6 and self.domain_extrema[0, 1] <= 7): Uf = torch.full(inputs.shape, -1)

        # Idle Phase
        elif (self.domain_extrema[0, 0] >= 1 and self.domain_extrema[0, 1] <= 2) or (self.domain_extrema[0, 0] >= 3 and self.domain_extrema[0, 1] <= 4) or (self.domain_extrema[0, 0] >= 5 and self.domain_extrema[0, 1] <= 6) or (self.domain_extrema[0, 0] >= 7 and self.domain_extrema[0, 1] <= 8): Uf = torch.full(inputs.shape, 0)

        return Uf
    
    ################################################################################################
    # Function returning the input-output tensor required to assemble the training set S_tb corresponding to the temporal boundary
    def add_temporal_boundary_points(self):
        t0 = self.domain_extrema[0, 0]
        input_tb = self.convert(self.soboleng.draw(self.n_tb))    # input_sb has two columns (t, x) both with random numbers in the two respective domains
        input_tb[:, 0] = torch.full(input_tb[:, 0].shape, t0)   # overwrite the entier column of time with t0
        output_tb = torch.full(input_tb[:, 0].shape, self.T_0).reshape(-1, 1)   # the output has 1 column

        return input_tb, output_tb  # input_tb is the sequence of x_n; output_tb is the sequence u0(x_n)

    # Function returning the input-output tensor required to assemble the training set S_sb corresponding to the spatial boundary
    def add_spatial_boundary_points(self):
        x0 = self.domain_extrema[1, 0]
        xL = self.domain_extrema[1, 1]

        # Dataset with random [t, x] in the domain
        input_sb = self.convert(self.soboleng.draw(self.n_sb))

        # Assigne the spatial boundary x=x0
        input_sb_0 = torch.clone(input_sb)
        input_sb_0[:, 1] = torch.full(input_sb_0[:, 1].shape, x0)

        # Assigne the spatial boundary x=xL
        input_sb_L = torch.clone(input_sb)
        input_sb_L[:, 1] = torch.full(input_sb_L[:, 1].shape, xL)

        # Require the gradient
        input_sb_0.requires_grad = True
        input_sb_L.requires_grad = True

        # Compute the output for the the different phases

        # Charging Phase
        if (self.domain_extrema[0, 1] <= 1) or (self.domain_extrema[0, 0] >= 4 and self.domain_extrema[0, 1] <=5 ):
            output_sb_0 = torch.full(input_sb_0[:, 0].shape, self.T_hot).reshape(-1, 1)
            output_sb_L = torch.full(input_sb_L[:, 0].shape, 0).reshape(-1, 1)
        
        # Discharging Phase
        elif (self.domain_extrema[0, 0] >= 2 and self.domain_extrema[0, 1] <= 3) or (self.domain_extrema[0, 0] >= 6 and self.domain_extrema[0, 1] <= 7):
            output_sb_0 = torch.full(input_sb_0[:, 0].shape, 0).reshape(-1, 1)
            output_sb_L = torch.full(input_sb_L[:, 0].shape, self.T_0).reshape(-1, 1)

        # Idle Phase
        elif (self.domain_extrema[0, 0] >= 1 and self.domain_extrema[0, 1] <= 2) or (self.domain_extrema[0, 0] >= 3 and self.domain_extrema[0, 1] <= 4) or (self.domain_extrema[0, 0] >= 5 and self.domain_extrema[0, 1] <= 6) or (self.domain_extrema[0, 0] >= 7 and self.domain_extrema[0, 1] <= 8):
            output_sb_0 = torch.full(input_sb_0[:, 0].shape, 0).reshape(-1, 1)
            output_sb_L = torch.full(input_sb_L[:, 0].shape, 0).reshape(-1, 1)               

        return torch.cat([input_sb_0, input_sb_L], dim=0), torch.cat([output_sb_0, output_sb_L], dim=0)


    # Function returning the input-output tensor required to assemble the training set S_int corresponding to the interior domain where the PDE is enforced
    def add_interior_points(self):
        # Now we use the convert fct since we want the t is in [0, 8]
        input_int = self.convert(self.soboleng.draw(self.n_int))
        output_int = torch.zeros((input_int.shape[0], 1))
        
        return input_int, output_int
    
    # Function returning the input-output tensor required to assemble the training set S_meas corresponding to the measured points in the domain.
    # These points are read from the file "DataSolution.txt"
    def get_measurement_data(self):
        # Read the CSV file into a DataFrame
        df_meas = pd.read_csv('DataSolution.txt')

        # From the data consider only the one in the specific phase the domain is in
        # Charge Phase 1
        if (self.domain_extrema[0, 1] <= 1):
            df_meas = df_meas[(df_meas['t']<=1)]
        
        # Charge Phase 2
        elif (self.domain_extrema[0, 0] >= 4 and self.domain_extrema[0, 1] <=5 ):
            df_meas = df_meas[(df_meas['t']>=4) & (df_meas['t']<=5)]
        
        # Discharge Phase 1
        elif (self.domain_extrema[0, 0] >= 2 and self.domain_extrema[0, 1] <= 3):
            df_meas = df_meas[(df_meas['t']>=2) & (df_meas['t']<=3)]
        
        # Discharge Phase 2
        elif (self.domain_extrema[0, 0] >= 6 and self.domain_extrema[0, 1] <= 7):
            df_meas = df_meas[(df_meas['t']>=6) & (df_meas['t']<=7)]
        
        # Idle Phase 1
        elif (self.domain_extrema[0, 0] >= 1 and self.domain_extrema[0, 1] <= 2):
            df_meas = df_meas[(df_meas['t']>=1) & (df_meas['t']<=2)]
        
        # Idle Phase 2
        elif (self.domain_extrema[0, 0] >= 3 and self.domain_extrema[0, 1] <= 4):
            df_meas = df_meas[(df_meas['t']>=3) & (df_meas['t']<=4)]

        # Idle Phase 3
        elif (self.domain_extrema[0, 0] >= 5 and self.domain_extrema[0, 1] <= 6):
            df_meas = df_meas[(df_meas['t']>=5) & (df_meas['t']<=6)]
        
        # Idle Phase 4
        elif (self.domain_extrema[0, 0] >= 7 and self.domain_extrema[0, 1] <= 8):
            df_meas = df_meas[(df_meas['t']>=7) & (df_meas['t']<=8)]

        # Convert the DataFrame to a torch.tensor
        tensor_meas = torch.tensor(df_meas.values , dtype=torch.float)
        
        # The first 2 columns are the inputs: [t, x]
        input_meas = tensor_meas[:, :2]
        
        # The last column are the outputs: [Tf]
        output_meas = tensor_meas[:, 2:]
        return input_meas, output_meas

    # Function returning the training sets S_sb, S_tb, S_int as dataloader
    def assemble_datasets(self):
        input_sb, output_sb = self.add_spatial_boundary_points()    # S_sb
        input_tb, output_tb = self.add_temporal_boundary_points()   # S_tb
        input_int, output_int = self.add_interior_points()          # S_int
        input_meas, output_meas = self.get_measurement_data()       # S_meas

        training_set_sb = DataLoader(torch.utils.data.TensorDataset(input_sb, output_sb), batch_size=2*self.space_dimensions*self.n_sb, shuffle=False)  #batch_size has *8 since there are 8 different phases and for each one we have 2 conditions (x0 & xL)
        training_set_tb = DataLoader(torch.utils.data.TensorDataset(input_tb, output_tb), batch_size=self.n_tb, shuffle=False)
        training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int, output_int), batch_size=self.n_int, shuffle=False)
        training_set_meas = DataLoader(torch.utils.data.TensorDataset(input_meas, output_meas), batch_size=output_meas.shape[0], shuffle=False)

        return training_set_sb, training_set_tb, training_set_int, training_set_meas

    ################################################################################################
    # Function to compute the terms required in the definition of the TEMPORAL boundary residual
    def apply_initial_condition(self, input_tb):
        u_pred_tb = self.approximate_solution(input_tb)
        
        return u_pred_tb

    # Function to compute the terms required in the definition of the SPATIAL boundary residual
    def apply_boundary_conditions(self, input_sb):
        assert (input_sb.requires_grad==True)   # make sure the grad is requested so we can compute the derivatives

        # Split the input into the 2 boundaries
        input_sb_0 = input_sb[:int(input_sb.shape[0]/2), :]
        input_sb_L = input_sb[int(input_sb.shape[0]/2):, :]
        
        # Charge Phase
        if (self.domain_extrema[0, 1] <= 1) or (self.domain_extrema[0, 0] >= 4 and self.domain_extrema[0, 1] <=5 ):
            # x0
            u_pred_sb_0 = self.approximate_solution(input_sb_0)

            # xL
            u_pred_Tf = self.approximate_solution(input_sb_L)
            u_pred_sb_L = torch.autograd.grad(u_pred_Tf.sum(), input_sb_L, create_graph=True)[0][:, 1].reshape(-1, 1) # take only d/dx
        
        # Discharge Phase
        elif (self.domain_extrema[0, 0] >= 2 and self.domain_extrema[0, 1] <= 3) or (self.domain_extrema[0, 0] >= 6 and self.domain_extrema[0, 1] <= 7):
            # x0
            u_pred_Tf = self.approximate_solution(input_sb_0)
            u_pred_sb_0 = torch.autograd.grad(u_pred_Tf.sum(), input_sb_0, create_graph=True)[0][:, 1].reshape(-1, 1) # take only d/dx

            # xL
            u_pred_sb_L = self.approximate_solution(input_sb_L)
        
        # Idle Phase
        elif (self.domain_extrema[0, 0] >= 1 and self.domain_extrema[0, 1] <= 2) or (self.domain_extrema[0, 0] >= 3 and self.domain_extrema[0, 1] <= 4) or (self.domain_extrema[0, 0] >= 5 and self.domain_extrema[0, 1] <= 6) or (self.domain_extrema[0, 0] >= 7 and self.domain_extrema[0, 1] <= 8):
            # x0
            u_pred_Tf = self.approximate_solution(input_sb_0)
            u_pred_sb_0 = torch.autograd.grad(u_pred_Tf.sum(), input_sb_0, create_graph=True)[0][:, 1].reshape(-1, 1) # take only d/dx

            # xL
            u_pred_Tf = self.approximate_solution(input_sb_L)
            u_pred_sb_L = torch.autograd.grad(u_pred_Tf.sum(), input_sb_L, create_graph=True)[0][:, 1].reshape(-1, 1) # take only d/dx
        
        return torch.cat([u_pred_sb_0, u_pred_sb_L], dim=0)

    # Function to compute the PDE residuals
    def compute_pde_residual(self, input_int):
        input_int.requires_grad = True
        u = self.approximate_solution(input_int).reshape(-1,)       # u is the solution (Tf) of the PDE
        g = self.approximate_coefficient(input_int).reshape(-1,)    # g is the function (Ts) that is requested

        # grad compute the gradient of a "SCALAR" function L with respect to some input nxm TENSOR Z=[[x1, y1],[x2,y2],[x3,y3],...,[xn,yn]], m=2
        # it returns grad_L = [[dL/dx1, dL/dy1],[dL/dx2, dL/dy2],[dL/dx3, dL/dy3],...,[dL/dxn, dL/dyn]]
        # Note: pytorch considers a tensor [u1, u2,u3, ... ,un] a vectorial function
        # whereas sum_u = u1 + u2 + u3 + u4 + ... + un as a "scalar" one

        # In our case ui = u(xi), therefore the line below returns:
        # grad_u = [[dsum_u/dx1, dsum_u/dy1],[dsum_u/dx2, dsum_u/dy2],[dsum_u/dx3, dL/dy3],...,[dsum_u/dxm, dsum_u/dyn]]
        # and dsum_u/dxi = d(u1 + u2 + u3 + u4 + ... + un)/dxi = d(u(x1) + u(x2) u3(x3) + u4(x4) + ... + u(xn))/dxi = dui/dxi

        # Since u for us is u = (uf, us), we have to devide the two cases

        grad_u = torch.autograd.grad(u.sum(), input_int, create_graph=True)[0]
        grad_u_t = grad_u[:, 0]
        grad_u_x = grad_u[:, 1]
        grad_u_xx = torch.autograd.grad(grad_u_x.sum(), input_int, create_graph=True)[0][:, 1]

        # Compute the velocity of the fluid Uf(t)
        Uf = self.fluid_velocity(input_int[:, 0])

        residual = (grad_u_t + Uf*grad_u_x) - (self.alpha_f*grad_u_xx - self.h_f*(u-g))

        return residual.reshape(-1, )

    # Function to compute the total loss (weighted sum of spatial boundary loss, temporal boundary loss and interior loss)
    def compute_loss(self, inp_train_sb, u_train_sb, inp_train_tb, u_train_tb, inp_train_int, inp_train_meas, u_train_meas, verbose=True):
        u_pred_sb = self.apply_boundary_conditions(inp_train_sb)
        u_pred_tb = self.apply_initial_condition(inp_train_tb)
        u_pred_meas = self.approximate_solution(inp_train_meas)

        assert (u_pred_sb.shape[1] == u_train_sb.shape[1])
        assert (u_pred_tb.shape[1] == u_train_tb.shape[1])
        assert (u_pred_meas.shape[1] == u_train_meas.shape[1])

        r_int = self.compute_pde_residual(inp_train_int)
        r_sb = u_train_sb - u_pred_sb
        r_tb = u_train_tb - u_pred_tb
        r_meas = u_train_meas - u_pred_meas

        loss_sb = torch.mean(abs(r_sb) ** 2)
        loss_tb = torch.mean(abs(r_tb) ** 2)
        loss_int = torch.mean(abs(r_int) ** 2)
        loss_meas = torch.mean(abs(r_meas) ** 2)

        loss_u = loss_sb + loss_tb + loss_meas

        loss = torch.log10(self.lambda_u * loss_u + loss_int)
        # loss = torch.log10(self.lambda_u * loss_u + loss_int)

        if verbose: print("Total loss: ", round(loss.item(), 4), "| Boundary Loss: ", round(torch.log10(loss_int).item(), 4), "| Measure Loss: ", round(torch.log10(loss_meas).item(), 4), "| Function Loss: ", round(torch.log10(loss_u).item(), 4))

        # Save the losses in some attributes
        self.total_loss = round(loss.item(), 4)
        self.boundary_loss = round(torch.log10(loss_int).item(), 4)
        self.measure_loss = round(torch.log10(loss_meas).item(), 4)
        self.function_loss = round(torch.log10(loss_u).item(), 4)

        return loss

    ################################################################################################
    def fit(self, num_epochs, optimizer, verbose=True):
        # Start the timer
        start_time = time.time()

        history = list()

        # Loop over epochs
        for epoch in range(num_epochs):
            if verbose: print("################################ ", epoch, " ################################")

            for j, ((inp_train_sb, u_train_sb), (inp_train_tb, u_train_tb), (inp_train_int, u_train_int), (inp_train_meas, u_train_meas)) in enumerate(zip(self.training_set_sb, self.training_set_tb, self.training_set_int, self.training_set_meas)):
                def closure():
                    optimizer.zero_grad()
                    loss = self.compute_loss(inp_train_sb, u_train_sb, inp_train_tb, u_train_tb, inp_train_int, inp_train_meas, u_train_meas, verbose=verbose)
                    loss.backward()

                    history.append(loss.item())
                    return loss
                
                optimizer.step(closure=closure)

        print('Final Loss: ', history[-1])
        print("Run time for the fitting: ", round((time.time()- start_time)/60), " minutes", round((time.time()- start_time)%60), " seconds")

        # Save the run time in an attribute
        self.run_time = (time.time()- start_time)
        return history

    ################################################################################################
    def plotting(self, vmax_=None, savefig=False):
        inputs = self.soboleng.draw(100000)
        inputs = self.convert(inputs)

        output_tf = self.approximate_solution(inputs).reshape(-1, )
        output_ts = self.approximate_coefficient(inputs).reshape(-1, )

        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        if vmax_ is None: im1 = axs[0].scatter(inputs[:, 0].detach(), inputs[:, 1].detach(), c=output_tf.detach(), cmap="jet")
        else: im1 = axs[0].scatter(inputs[:, 0].detach(), inputs[:, 1].detach(), c=output_tf.detach(), cmap="jet", vmax=vmax_)
        axs[0].set_xlabel("t")
        axs[0].set_ylabel("x")
        plt.colorbar(im1, ax=axs[0])
        axs[0].grid(True, which="both", ls=":")
        if vmax_ is None: im2 = axs[1].scatter(inputs[:, 0].detach(), inputs[:, 1].detach(), c=output_ts.detach(), cmap="jet")
        else: im2 = axs[1].scatter(inputs[:, 0].detach(), inputs[:, 1].detach(), c=output_ts.detach(), cmap="jet", vmax=vmax_)
        axs[1].set_xlabel("t")
        axs[1].set_ylabel("x")
        plt.colorbar(im2, ax=axs[1])
        axs[1].grid(True, which="both", ls=":")
        axs[0].set_title(r"Approximate Solution $T_f$")
        axs[1].set_title(r"Approximate Solution $T_s$")

        fig.suptitle(f"lambda_u = {self.lambda_u} neurons = {self.coefficient_neurons}\nn_int = {self.n_int} n_sb = {self.n_sb} n_tb = {self.n_tb}", size=18)
        
        if savefig :plt.savefig(f"Plots/lambda_u={self.lambda_u}_neurons={self.coefficient_neurons}_n_int={self.n_int}_n_sb={self.n_sb}_n_tb={self.n_tb}.png")
        
        plt.show()

class Pinns_Cycle_all:
    """
    Class to define the Physics Informed Neural Network (PINN) for the task 2 of the project.
    
    With this class we are going to train the PINN on a specific cycle,
    the idea here to to split the different phases of the cycles and 3 different NN, one on each phase.
    """
    def __init__(self, n_int_, n_sb_, n_tb_, t0_=0, tf_=8, lambda_u_=10, coefficient_neurons_=20, coefficient_layers_=4, alpha_f_=0.005, h_f_=5, T_hot_=4, T_0_=1, U_f_=None):
        self.n_int = n_int_     # n_int_:= number of intertior points
        self.n_sb = n_sb_       # n_sb_ := number of spatial boundary points
        self.n_tb = n_tb_       # n_tb_ := number of time boundary points

        # Set the paremeters of the equation
        self.alpha_f = alpha_f_
        self.h_f = h_f_
        self.T_hot = T_hot_
        self.T_0 = T_0_
        self.U_f = U_f_

        # Extrema of the solution domain (t,x) in [0, t]x[0, L]
        self.domain_extrema = torch.tensor([[t0_, tf_],  # Time dimension
                                            [0, 1]])  # Space dimension

        # Number of space dimensions
        self.space_dimensions = 1

        # Parameter to balance role of data and PDE
        self.lambda_u = lambda_u_

        # Number of neurons in the coefficient NN
        self.coefficient_neurons =coefficient_neurons_
        self.coefficient_layers = coefficient_layers_

        # FF Dense NN to approximate the solution of the underlying reaction-convection-diffusion equations of the fluid
        self.approximate_charging_solution = NeuralNet(input_dimension=self.domain_extrema.shape[0], output_dimension=1, # is a NN with input_dim=2 (time & space), output_dim=1 (fluid_temp)
                                              n_hidden_layers=4,
                                              neurons=20,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=42)
        
        self.approximate_discharging_solution = NeuralNet(input_dimension=self.domain_extrema.shape[0], output_dimension=1, # is a NN with input_dim=2 (time & space), output_dim=1 (fluid_temp)
                                              n_hidden_layers=4,
                                              neurons=20,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=42)
        
        self.approximate_idle_solution = NeuralNet(input_dimension=self.domain_extrema.shape[0], output_dimension=1, # is a NN with input_dim=2 (time & space), output_dim=1 (fluid_temp)
                                              n_hidden_layers=4,
                                              neurons=20,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=42)
        # FF Dense NN to approximate the solid temperature we wish to infer
        self.approximate_coefficient = NeuralNet(input_dimension=self.domain_extrema.shape[0], output_dimension=1,  # is a NN with input_dim=2 (time & space), output_dim=1 (solid_temp)
                                                 n_hidden_layers=coefficient_layers_,
                                                 neurons=coefficient_neurons_,
                                                 regularization_param=0.,
                                                 regularization_exp=2.,
                                                 retrain_seed=42)

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])   # it will create a 2 cloumns tensor, the rows nunmber is specified after every time it is used

        # Training sets S_sb, S_tb, S_int as torch dataloader
        self.training_set_sb, self.training_set_tb, self.training_set_int, self.training_set_meas = self.assemble_datasets()

    ################################################################################################
    # Function to linearly transform a tensor whose value are between 0 and 1
    # to a tensor whose values are between the domain extrema
    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]
    
    # Function Uf(t) -> given the time it gives back the velocity of the fluid in the relative phase
    def fluid_velocity(self, inputs):
        Uf = torch.full(inputs.shape, 999)  # give all 999 for semplicity when testing

        for i, t in enumerate(inputs):
            # Charging Phase
            if (t <= 1) or (t > 4 and t <=5 ): Uf[i] = 1
            # Discharging Phase
            elif (t > 2 and t <= 3) or (t > 6 and t <= 7): Uf[i] = -1
            # Idle Phase
            elif (t > 1 and t <= 2) or (t > 3 and t <= 4) or (t > 5 and t <= 6) or (t > 7 and t <= 8): Uf[i] = 0

        return Uf
    
    ################################################################################################
    # Function returning the input-output tensor required to assemble the training set S_tb corresponding to the temporal boundary
    def add_temporal_boundary_points(self):
        t0 = self.domain_extrema[0, 0]
        input_tb = self.convert(self.soboleng.draw(self.n_tb))    # input_sb has two columns (t, x) both with random numbers in the two respective domains
        input_tb[:, 0] = torch.full(input_tb[:, 0].shape, t0)   # overwrite the entier column of time with t0
        output_tb = torch.full(input_tb[:, 0].shape, self.T_0).reshape(-1, 1)   # the output has 1 column

        return input_tb, output_tb  # input_tb is the sequence of x_n; output_tb is the sequence u0(x_n)

    # Function returning the input-output tensor required to assemble the training set S_sb corresponding to the spatial boundary
    def add_spatial_boundary_points(self):
        x0 = self.domain_extrema[1, 0]
        xL = self.domain_extrema[1, 1]

        # Dataset with random [t, x] in the domain
        input_sb = self.convert(self.soboleng.draw(self.n_sb))

        # Assigne the spatial boundary x=x0
        input_sb_0 = torch.clone(input_sb)
        input_sb_0[:, 1] = torch.full(input_sb_0[:, 1].shape, x0)

        # Assigne the spatial boundary x=xL
        input_sb_L = torch.clone(input_sb)
        input_sb_L[:, 1] = torch.full(input_sb_L[:, 1].shape, xL)

        # Require the gradient
        input_sb_0.requires_grad = True
        input_sb_L.requires_grad = True

        # Compute the output for the the different phases

        # Charging Phase
        if (self.domain_extrema[0, 1] <= 1) or (self.domain_extrema[0, 0] >= 4 and self.domain_extrema[0, 1] <=5 ):
            output_sb_0 = torch.full(input_sb_0[:, 0].shape, self.T_hot).reshape(-1, 1)
            output_sb_L = torch.full(input_sb_L[:, 0].shape, 0).reshape(-1, 1)
        
        # Discharging Phase
        elif (self.domain_extrema[0, 0] >= 2 and self.domain_extrema[0, 1] <= 3) or (self.domain_extrema[0, 0] >= 6 and self.domain_extrema[0, 1] <= 7):
            output_sb_0 = torch.full(input_sb_0[:, 0].shape, 0).reshape(-1, 1)
            output_sb_L = torch.full(input_sb_L[:, 0].shape, self.T_0).reshape(-1, 1)

        # Idle Phase
        elif (self.domain_extrema[0, 0] >= 1 and self.domain_extrema[0, 1] <= 2) or (self.domain_extrema[0, 0] >= 3 and self.domain_extrema[0, 1] <= 4) or (self.domain_extrema[0, 0] >= 5 and self.domain_extrema[0, 1] <= 6) or (self.domain_extrema[0, 0] >= 7 and self.domain_extrema[0, 1] <= 8):
            output_sb_0 = torch.full(input_sb_0[:, 0].shape, 0).reshape(-1, 1)
            output_sb_L = torch.full(input_sb_L[:, 0].shape, 0).reshape(-1, 1)               

        return torch.cat([input_sb_0, input_sb_L], dim=0), torch.cat([output_sb_0, output_sb_L], dim=0)


    # Function returning the input-output tensor required to assemble the training set S_int corresponding to the interior domain where the PDE is enforced
    def add_interior_points(self):
        # Now we use the convert fct since we want the t is in [0, 8]
        input_int = self.convert(self.soboleng.draw(self.n_int))
        output_int = torch.zeros((input_int.shape[0], 1))
        
        return input_int, output_int
    
    # Function returning the input-output tensor required to assemble the training set S_meas corresponding to the measured points in the domain.
    # These points are read from the file "DataSolution.txt"
    def get_measurement_data(self):
        # Read the CSV file into a DataFrame
        df_meas = pd.read_csv('DataSolution.txt')

        # From the data consider only the one in the specific phase the domain is in
        # Charge Phase 1
        if (self.domain_extrema[0, 1] <= 1):
            df_meas = df_meas[(df_meas['t']<=1)]
        
        # Charge Phase 2
        elif (self.domain_extrema[0, 0] >= 4 and self.domain_extrema[0, 1] <=5 ):
            df_meas = df_meas[(df_meas['t']>=4) & (df_meas['t']<=5)]
        
        # Discharge Phase 1
        elif (self.domain_extrema[0, 0] >= 2 and self.domain_extrema[0, 1] <= 3):
            df_meas = df_meas[(df_meas['t']>=2) & (df_meas['t']<=3)]
        
        # Discharge Phase 2
        elif (self.domain_extrema[0, 0] >= 6 and self.domain_extrema[0, 1] <= 7):
            df_meas = df_meas[(df_meas['t']>=6) & (df_meas['t']<=7)]
        
        # Idle Phase 1
        elif (self.domain_extrema[0, 0] >= 1 and self.domain_extrema[0, 1] <= 2):
            df_meas = df_meas[(df_meas['t']>=1) & (df_meas['t']<=2)]
        
        # Idle Phase 2
        elif (self.domain_extrema[0, 0] >= 3 and self.domain_extrema[0, 1] <= 4):
            df_meas = df_meas[(df_meas['t']>=3) & (df_meas['t']<=4)]

        # Idle Phase 3
        elif (self.domain_extrema[0, 0] >= 5 and self.domain_extrema[0, 1] <= 6):
            df_meas = df_meas[(df_meas['t']>=5) & (df_meas['t']<=6)]
        
        # Idle Phase 4
        elif (self.domain_extrema[0, 0] >= 7 and self.domain_extrema[0, 1] <= 8):
            df_meas = df_meas[(df_meas['t']>=7) & (df_meas['t']<=8)]

        # Convert the DataFrame to a torch.tensor
        tensor_meas = torch.tensor(df_meas.values , dtype=torch.float)
        
        # The first 2 columns are the inputs: [t, x]
        input_meas = tensor_meas[:, :2]
        
        # The last column are the outputs: [Tf]
        output_meas = tensor_meas[:, 2:]
        return input_meas, output_meas

    # Function returning the training sets S_sb, S_tb, S_int as dataloader
    def assemble_datasets(self):
        input_sb, output_sb = self.add_spatial_boundary_points()    # S_sb
        input_tb, output_tb = self.add_temporal_boundary_points()   # S_tb
        input_int, output_int = self.add_interior_points()          # S_int
        input_meas, output_meas = self.get_measurement_data()       # S_meas

        training_set_sb = DataLoader(torch.utils.data.TensorDataset(input_sb, output_sb), batch_size=2*self.space_dimensions*self.n_sb, shuffle=False)  #batch_size has *8 since there are 8 different phases and for each one we have 2 conditions (x0 & xL)
        training_set_tb = DataLoader(torch.utils.data.TensorDataset(input_tb, output_tb), batch_size=self.n_tb, shuffle=False)
        training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int, output_int), batch_size=self.n_int, shuffle=False)
        training_set_meas = DataLoader(torch.utils.data.TensorDataset(input_meas, output_meas), batch_size=output_meas.shape[0], shuffle=False)

        return training_set_sb, training_set_tb, training_set_int, training_set_meas

    ################################################################################################
    # Function to compute the terms required in the definition of the TEMPORAL boundary residual
    def apply_initial_condition(self, input_tb):
        u_pred_tb = self.approximate_solution(input_tb)
        
        return u_pred_tb

    # Function to compute the terms required in the definition of the SPATIAL boundary residual
    def apply_boundary_conditions(self, input_sb):
        assert (input_sb.requires_grad==True)   # make sure the grad is requested so we can compute the derivatives

        # Split the input into the 2 boundaries
        input_sb_0 = input_sb[:int(input_sb.shape[0]/2), :]
        input_sb_L = input_sb[int(input_sb.shape[0]/2):, :]
        
        # Charge Phase
        if (self.domain_extrema[0, 1] <= 1) or (self.domain_extrema[0, 0] >= 4 and self.domain_extrema[0, 1] <=5 ):
            # x0
            u_pred_sb_0 = self.approximate_solution(input_sb_0)

            # xL
            u_pred_Tf = self.approximate_solution(input_sb_L)
            u_pred_sb_L = torch.autograd.grad(u_pred_Tf.sum(), input_sb_L, create_graph=True)[0][:, 1].reshape(-1, 1) # take only d/dx
        
        # Discharge Phase
        elif (self.domain_extrema[0, 0] >= 2 and self.domain_extrema[0, 1] <= 3) or (self.domain_extrema[0, 0] >= 6 and self.domain_extrema[0, 1] <= 7):
            # x0
            u_pred_Tf = self.approximate_solution(input_sb_0)
            u_pred_sb_0 = torch.autograd.grad(u_pred_Tf.sum(), input_sb_0, create_graph=True)[0][:, 1].reshape(-1, 1) # take only d/dx

            # xL
            u_pred_sb_L = self.approximate_solution(input_sb_L)
        
        # Idle Phase
        elif (self.domain_extrema[0, 0] >= 1 and self.domain_extrema[0, 1] <= 2) or (self.domain_extrema[0, 0] >= 3 and self.domain_extrema[0, 1] <= 4) or (self.domain_extrema[0, 0] >= 5 and self.domain_extrema[0, 1] <= 6) or (self.domain_extrema[0, 0] >= 7 and self.domain_extrema[0, 1] <= 8):
            # x0
            u_pred_Tf = self.approximate_solution(input_sb_0)
            u_pred_sb_0 = torch.autograd.grad(u_pred_Tf.sum(), input_sb_0, create_graph=True)[0][:, 1].reshape(-1, 1) # take only d/dx

            # xL
            u_pred_Tf = self.approximate_solution(input_sb_L)
            u_pred_sb_L = torch.autograd.grad(u_pred_Tf.sum(), input_sb_L, create_graph=True)[0][:, 1].reshape(-1, 1) # take only d/dx
        
        return torch.cat([u_pred_sb_0, u_pred_sb_L], dim=0)

    # Function to compute the PDE residuals
    def compute_pde_residual(self, input_int):
        input_int.requires_grad = True
        u = self.approximate_solution(input_int).reshape(-1,)       # u is the solution (Tf) of the PDE
        g = self.approximate_coefficient(input_int).reshape(-1,)    # g is the function (Ts) that is requested

        # grad compute the gradient of a "SCALAR" function L with respect to some input nxm TENSOR Z=[[x1, y1],[x2,y2],[x3,y3],...,[xn,yn]], m=2
        # it returns grad_L = [[dL/dx1, dL/dy1],[dL/dx2, dL/dy2],[dL/dx3, dL/dy3],...,[dL/dxn, dL/dyn]]
        # Note: pytorch considers a tensor [u1, u2,u3, ... ,un] a vectorial function
        # whereas sum_u = u1 + u2 + u3 + u4 + ... + un as a "scalar" one

        # In our case ui = u(xi), therefore the line below returns:
        # grad_u = [[dsum_u/dx1, dsum_u/dy1],[dsum_u/dx2, dsum_u/dy2],[dsum_u/dx3, dL/dy3],...,[dsum_u/dxm, dsum_u/dyn]]
        # and dsum_u/dxi = d(u1 + u2 + u3 + u4 + ... + un)/dxi = d(u(x1) + u(x2) u3(x3) + u4(x4) + ... + u(xn))/dxi = dui/dxi

        # Since u for us is u = (uf, us), we have to devide the two cases

        grad_u = torch.autograd.grad(u.sum(), input_int, create_graph=True)[0]
        grad_u_t = grad_u[:, 0]
        grad_u_x = grad_u[:, 1]
        grad_u_xx = torch.autograd.grad(grad_u_x.sum(), input_int, create_graph=True)[0][:, 1]

        # Compute the velocity of the fluid Uf(t)
        Uf = self.fluid_velocity(input_int[:, 0])

        residual = (grad_u_t + Uf*grad_u_x) - (self.alpha_f*grad_u_xx - self.h_f*(u-g))

        return residual.reshape(-1, )

    # Function to compute the total loss (weighted sum of spatial boundary loss, temporal boundary loss and interior loss)
    def compute_loss(self, inp_train_sb, u_train_sb, inp_train_tb, u_train_tb, inp_train_int, inp_train_meas, u_train_meas, verbose=True):
        u_pred_sb = self.apply_boundary_conditions(inp_train_sb)
        u_pred_tb = self.apply_initial_condition(inp_train_tb)
        u_pred_meas = self.approximate_solution(inp_train_meas)

        assert (u_pred_sb.shape[1] == u_train_sb.shape[1])
        assert (u_pred_tb.shape[1] == u_train_tb.shape[1])
        assert (u_pred_meas.shape[1] == u_train_meas.shape[1])

        r_int = self.compute_pde_residual(inp_train_int)
        r_sb = u_train_sb - u_pred_sb
        r_tb = u_train_tb - u_pred_tb
        r_meas = u_train_meas - u_pred_meas

        loss_sb = torch.mean(abs(r_sb) ** 2)
        loss_tb = torch.mean(abs(r_tb) ** 2)
        loss_int = torch.mean(abs(r_int) ** 2)
        loss_meas = torch.mean(abs(r_meas) ** 2)

        loss_u = loss_sb + loss_tb + loss_meas

        loss = torch.log10(self.lambda_u * loss_u + loss_int)
        # loss = torch.log10(self.lambda_u * loss_u + loss_int)

        if verbose: print("Total loss: ", round(loss.item(), 4), "| Boundary Loss: ", round(torch.log10(loss_int).item(), 4), "| Measure Loss: ", round(torch.log10(loss_meas).item(), 4), "| Function Loss: ", round(torch.log10(loss_u).item(), 4))

        # Save the losses in some attributes
        self.total_loss = round(loss.item(), 4)
        self.boundary_loss = round(torch.log10(loss_int).item(), 4)
        self.measure_loss = round(torch.log10(loss_meas).item(), 4)
        self.function_loss = round(torch.log10(loss_u).item(), 4)

        return loss

    ################################################################################################
    def fit(self, num_epochs, optimizer, verbose=True):
        # Start the timer
        start_time = time.time()

        history = list()

        # Loop over epochs
        for epoch in range(num_epochs):
            if verbose: print("################################ ", epoch, " ################################")

            for j, ((inp_train_sb, u_train_sb), (inp_train_tb, u_train_tb), (inp_train_int, u_train_int), (inp_train_meas, u_train_meas)) in enumerate(zip(self.training_set_sb, self.training_set_tb, self.training_set_int, self.training_set_meas)):
                def closure():
                    optimizer.zero_grad()
                    loss = self.compute_loss(inp_train_sb, u_train_sb, inp_train_tb, u_train_tb, inp_train_int, inp_train_meas, u_train_meas, verbose=verbose)
                    loss.backward()

                    history.append(loss.item())
                    return loss
                
                optimizer.step(closure=closure)

        print('Final Loss: ', history[-1])
        print("Run time for the fitting: ", round((time.time()- start_time)/60), " minutes", round((time.time()- start_time)%60), " seconds")

        # Save the run time in an attribute
        self.run_time = (time.time()- start_time)
        return history

    ################################################################################################
    def plotting(self, vmax_=None, savefig=False):
        inputs = self.soboleng.draw(100000)
        inputs = self.convert(inputs)

        output_tf = self.approximate_solution(inputs).reshape(-1, )
        output_ts = self.approximate_coefficient(inputs).reshape(-1, )

        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        if vmax_ is None: im1 = axs[0].scatter(inputs[:, 0].detach(), inputs[:, 1].detach(), c=output_tf.detach(), cmap="jet")
        else: im1 = axs[0].scatter(inputs[:, 0].detach(), inputs[:, 1].detach(), c=output_tf.detach(), cmap="jet", vmax=vmax_)
        axs[0].set_xlabel("t")
        axs[0].set_ylabel("x")
        plt.colorbar(im1, ax=axs[0])
        axs[0].grid(True, which="both", ls=":")
        if vmax_ is None: im2 = axs[1].scatter(inputs[:, 0].detach(), inputs[:, 1].detach(), c=output_ts.detach(), cmap="jet")
        else: im2 = axs[1].scatter(inputs[:, 0].detach(), inputs[:, 1].detach(), c=output_ts.detach(), cmap="jet", vmax=vmax_)
        axs[1].set_xlabel("t")
        axs[1].set_ylabel("x")
        plt.colorbar(im2, ax=axs[1])
        axs[1].grid(True, which="both", ls=":")
        axs[0].set_title(r"Approximate Solution $T_f$")
        axs[1].set_title(r"Approximate Solution $T_s$")

        fig.suptitle(f"lambda_u = {self.lambda_u} neurons = {self.coefficient_neurons}\nn_int = {self.n_int} n_sb = {self.n_sb} n_tb = {self.n_tb}", size=18)
        
        if savefig :plt.savefig(f"Plots/lambda_u={self.lambda_u}_neurons={self.coefficient_neurons}_n_int={self.n_int}_n_sb={self.n_sb}_n_tb={self.n_tb}.png")
        
        plt.show()