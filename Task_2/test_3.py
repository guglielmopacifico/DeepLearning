from Pinns_task2 import *
import os

"""=================================================================================================================="""
# Define the model
n_int = 256
n_sb = 64
n_tb = 64

coefficient_neurons = 30

pinn = Pinns(n_int, n_sb, n_tb, coefficient_neurons_=coefficient_neurons)

n_epochs = 1
optimizer_LBFGS = optim.LBFGS(list(pinn.approximate_solution.parameters()) + list(pinn.approximate_coefficient.parameters()),
                              lr=float(0.5),
                              max_iter=50000,
                              max_eval=50000,
                              history_size=150,
                              line_search_fn="strong_wolfe",
                              tolerance_change=1.0 * np.finfo(float).eps)

hist = pinn.fit(num_epochs=n_epochs,
                optimizer=optimizer_LBFGS,
                verbose=False)

# Make a folder where to save the results
path = f'Outputs/N_30'
os.makedirs(path)

# Save the parameters
torch.save(pinn.approximate_solution.state_dict(), f'{path}/approximate_solution_parameters.pth')
torch.save(pinn.approximate_coefficient.state_dict(), f'{path}/approximate_coefficient_parameters.pth')

# Save the loss history
np.savetxt(f'{path}/loss_history.txt', np.array(hist))

# Make a file INFO_RUN where the parameters used for the run are saved
with open(f'{path}/INFO_RUN.txt', 'w') as f:

    f.write("Netowrk parameters:")
    f.write(f"\n\tlambda_u = {pinn.lambda_u}")
    f.write(f"\n\tcoefficient_neurons = {pinn.coefficient_neurons}")

    f.write(f"\n\Output Obtained:")
    f.write(f"\n\n\ttotal loss: = {pinn.total_loss}")
    f.write(f"\n\tbounday loss = {pinn.boundary_loss}")
    f.write(f"\n\tfunction loss = {pinn.function_loss}")
    f.write(f"\n\tmeasure loss = {pinn.measure_loss}")
    
    f.write(f"\n\nThis run took: {pinn.run_time/60} minutes {pinn.run_time%60} seconds")