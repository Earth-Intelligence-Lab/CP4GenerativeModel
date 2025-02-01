import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class GaussianPath:
    def __init__(self, alpha_fn, beta_fn):
        self.alpha = alpha_fn  # A function defining sigma(t)
        self.beta = beta_fn

    def sample_conditional_path(self, z, t):
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        x_t = alpha_t * z + beta_t * torch.randn_like(z)
        return x_t

    def conditional_vector_field(self, x, z, t):
        cvf = (self.alpha.dt(t) - (self.beta.dt(t) / self.beta(t)) * self.alpha(t)) * z + (self.beta.dt(t) / self.beta(t)) * x
        return cvf


class LinearAlpha():
    def __init__(self):
        pass

    def __call__(self, t):
        alpha_t = t
        return alpha_t

    def dt(self, t):
        return torch.ones_like(t)


class SquareRootBeta():
    def __init__(self):
        pass

    def __call__(self, t):
        beta_t = torch.sqrt(1 - t)
        return beta_t

    def dt(self, t):
        return - 0.5 / (torch.sqrt(1 - t) + 1e-4)


class FlowMatchingNet(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + condition_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, input_dim)

    def forward(self, y, condition, t):
        t = t.view(-1, 1)  # Ensure t has the right shape
        x = torch.cat([y, condition, t], dim=-1)  # Concatenate y, condition, and time
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)


def flow_matching_loss(model, gaussian_path, z, condition, t, device):

    x_t = gaussian_path.sample_conditional_path(z, t)
    cvf = gaussian_path.conditional_vector_field(x_t, z, t)
    vf_pred = model(x_t, condition, t)

    # MSE loss
    return F.mse_loss(vf_pred, cvf)


def train_flow_matching(model, gaussian_path, dataloader, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        Loss = 0  # Initialize loss for this epoch

        for batch in dataloader:
            condition, z = batch  # Original data and condition
            condition, z = condition.to(device), z.to(device)

            # Sample random time t
            t = torch.rand((z.size(0), 1), device=device)  # Uniformly sample t in [0, 1]

            # Compute loss
            loss = flow_matching_loss(model, gaussian_path, z, condition, t, device)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            Loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {Loss:.4f}")


def generate_data(model, gaussian_path, condition, timesteps, device):
    # Start from Gaussian noise
    x = torch.randn(condition.size(0), model.fc5.out_features).to(device)
    t_space = torch.linspace(0, 1, timesteps, device=device)

    for t in t_space:
        with torch.no_grad():
            dt = 1 / (timesteps - 1)
            u_t = model(x, condition, t * torch.ones(condition.size(0), 1).to(device))
            x = x + u_t * dt  # Euler step

    return x


def generate_samples_for_dataset(model, gaussian_path, data_loader, num_samples, timesteps, device):
    model.eval()  # Set the model to evaluation mode
    all_generated_samples = []  # To store all generated samples
    conditions = []  # To store corresponding conditions

    for condition_batch in data_loader:
        X = condition_batch[0].to(device)  # Get condition inputs from calib_loader

        # Generate num_samples_per_condition for each condition in the batch
        for condition in X:
            condition = condition.unsqueeze(0).repeat(num_samples, 1)  # Repeat condition
            generated_samples = generate_data(model, gaussian_path, condition, timesteps, device)
            all_generated_samples.append(generated_samples.cpu().numpy())
            conditions.append(condition.cpu().numpy())

    # Convert lists to arrays for easier handling
    all_generated_samples = np.array(all_generated_samples)
    conditions = np.array(conditions)
    return all_generated_samples, conditions
