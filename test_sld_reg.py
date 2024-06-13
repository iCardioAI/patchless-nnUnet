import torch

# Example tensor for coordinates
coords = torch.randn(1, 100, 2, 2)
sld = torch.zeros_like(coords)

winlen = 4
# Squeeze the coordinates for easier manipulation
coords_squeezed = coords.squeeze(0)

# Create indices for the sliding windows
indices = torch.arange(coords.shape[1])
index_matrix = indices.expand(coords.shape[1], coords.shape[1])

# Create masks to handle the window boundaries
mask = (index_matrix >= indices.unsqueeze(1) - winlen) & (index_matrix <= indices.unsqueeze(1) + winlen)

# Mask out-of-bound indices
masked_coords = coords_squeezed.unsqueeze(0).expand(coords.shape[1], *coords.shape[1:]).clone()
masked_coords[~mask] = 0

# Count the valid elements in each window for proper averaging
count_valid = mask.sum(dim=1, keepdim=True).float()

# Compute the mean of the sliding windows
mean_values = masked_coords.sum(dim=1) / count_valid

# Compute the squared difference from the mean
sld[0] = (mean_values - coords_squeezed) ** 2