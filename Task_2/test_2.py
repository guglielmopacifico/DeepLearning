from Pinns_task2 import *
import os

"""=================================================================================================================="""
# Define the model
n_int = 256*4
n_sb = 64*2
n_tb = 64*2


lambda_u = 5
coefficient_neurons = 30

pinn = Pinns(n_int, n_sb, n_tb, lambda_u_=lambda_u, coefficient_neurons_=coefficient_neurons)

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
path = f'Outputs/Test_6'
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
    f.write(f"\n\tn_int = {pinn.n_int}")
    f.write(f"\n\tn_sb = {pinn.n_sb}")
    f.write(f"\n\tn_tb = {pinn.n_tb}")

    f.write(f"\nOutput Obtained:")
    f.write(f"\n\n\ttotal loss:\t= {round(pinn.total_loss, 4)}")
    f.write(f"\n\tbounday loss\t= {round(pinn.boundary_loss, 4)}")
    f.write(f"\n\tfunction loss\t= {round(pinn.function_loss, 4)}")
    f.write(f"\n\tmeasure loss\t= {round(pinn.measure_loss, 4)}")
    
    f.write(f"\n\nThis run took: {round(pinn.run_time/60)} minutes {round(pinn.run_time%60)} seconds")