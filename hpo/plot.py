import gpytorch
from gpytorch.kernels import RBFKernel
from gpytorch.models import ExactGP
import torch
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
likelihood.noise = 0.1

outputscale = 1
lengthscale = 1

k_ard = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2))
k_ard.outputscale = 1
k_ard.base_kernel.lengthscale = [0.1, 5]

kernel = k_ard

with open("hpo_results_gamma_pos/bayesian_optimization2.json") as f:
    hpo_framework_results = json.loads(f.read())

df = pd.DataFrame(hpo_framework_results)
df_expanded = pd.json_normalize(df[0])
df = pd.concat([df, df_expanded], axis=1)

x_features = ['mdp.gamma', 'mdp.reward.position']
x = df[x_features].to_numpy()
y = df[1].to_numpy().reshape(-1, 1)



train_x = torch.from_numpy(x - x.mean())
train_y = torch.from_numpy((y - y.mean()).ravel())

fig = plt.figure(figsize=(20, 10))
plt.suptitle(f"")
mlls = {}
model = ExactGPModel(train_x, train_y, likelihood, kernel).eval()
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 100
for training_i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    if training_i % 20 == 0:
        print('Iter %d/%d - Loss: %.3f  outputscale: %.3f lengthscale: %.20s   noise: %.3f' % (
            training_i + 1, training_iter, loss.item(),
            model.covar_module.outputscale.item(),
            str(model.covar_module.base_kernel.lengthscale.detach().numpy()),
            model.likelihood.noise.item()
        ))
    optimizer.step()
current_mll = mll(model(train_x), train_y)
# print(f"Marginal Log Likelihood ({kernel_name}): {current_mll}")
# mlls[kernel_name] = current_mll.item()
model.eval()
    
x1_grid, x2_grid = np.meshgrid(np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 100), np.linspace(np.min(x[:, 1]), np.max(x[:, 1]), 100))
x_test = torch.from_numpy(np.vstack([x1_grid.ravel() - x[:, 0].mean(), x2_grid.ravel() - x[:, 1].mean()]).T)
with torch.no_grad():
    predictive_dist = model(x_test)
z_test = predictive_dist.mean.numpy().reshape(x1_grid.shape) + y.mean()
ax = fig.add_subplot(1, 1, 1, projection='3d')
surf = ax.plot_surface(x1_grid, x2_grid, z_test, cmap='viridis')
elev, azim = ax.elev, ax.azim
new_azimuth = azim - 90
ax.view_init(elev=elev, azim=new_azimuth)
ax.set_xlabel(x_features[0])
ax.set_ylabel(x_features[1])
ax.set_zlabel('Objective')
# ax.set_title(kernel_name)
ax.scatter(x[:, 0], x[:, 1], y.ravel())

plt.show()