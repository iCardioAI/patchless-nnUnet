import torch
torch.manual_seed(12345)
from torch import nn
import dsntnn

class FCN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.layers(x)

class CoordRegressionNetwork(nn.Module):
    def __init__(self, n_locations):
        super().__init__()
        self.fcn = FCN()
        self.hm_conv = nn.Conv2d(16, n_locations, kernel_size=1, bias=False)

    def forward(self, images):
        # 1. Run the images through our FCN
        fcn_out = self.fcn(images)
        # 2. Use a 1x1 conv to get one unnormalized heatmap per location
        unnormalized_heatmaps = self.hm_conv(fcn_out)
        # 3. Normalize the heatmaps
        heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps)
        # plt.figure()
        # plt.imshow(unnormalized_heatmaps[0, 0, :, :].cpu().detach().numpy())
        # plt.figure()
        # plt.imshow(heatmaps[0, 0, :, :].cpu().detach().numpy())

        # from patchless_nnunet.utils.custom_dsnt import flat_softmax as fs
        # rep = unnormalized_heatmaps[..., None].repeat(1, 1, 1, 1, 4)
        # plt.figure()
        # plt.imshow(rep[0, 0, :, :, 0].cpu().detach().numpy())
        # urep = fs(rep)
        # plt.figure()
        # plt.imshow(urep[0, 0, :, :, 0].cpu().detach().numpy())
        # plt.show()


        # 4. Calculate the coordinates
        # coords = dsntnn.dsnt(urep[..., 0])
        coords = dsntnn.dsnt(heatmaps)
        return coords, heatmaps


from torch import optim
import matplotlib.pyplot as plt
import scipy.misc
from skimage import transform
image_size = [40, 40]
raccoon_face = transform.resize(scipy.misc.face()[200:400, 600:800, :], image_size)
eye_x, eye_y = [2, 24], [20, 26]

# plt.imshow(raccoon_face)
# plt.scatter([eye_x], [eye_y], color='red', marker='X')
# plt.show()


raccoon_face_tensor = torch.from_numpy(raccoon_face).permute(2, 0, 1).float()
input_tensor = raccoon_face_tensor.div(255).unsqueeze(0).repeat(16, 1, 1, 1)
input_var = input_tensor.cuda()

eye_coords_tensor = torch.Tensor([[eye_x, eye_y]]).repeat(16, 1, 1)
target_tensor = (eye_coords_tensor * 2 + 1) / torch.Tensor(image_size) - 1
target_var = target_tensor.cuda()

# print('Target: {:0.4f}, {:0.4f}'.format(*list(target_tensor.squeeze())))

model = CoordRegressionNetwork(n_locations=2).cuda()
coords, heatmaps = model(input_var)

print('Initial prediction: {:0.4f}, {:0.4f}'.format(*list(coords[0, 0])))
# plt.imshow(heatmaps[0, 0].detach().cpu().numpy())
# plt.show()

optimizer = optim.RMSprop(model.parameters(), lr=2.5e-4)
mse = torch.nn.MSELoss(reduction='mean')
losses = []
euc = []
reg = []
for i in range(100):
    # Forward pass
    coords, heatmaps = model(input_var)

    # Per-location euclidean losses
    euc_losses = dsntnn.euclidean_losses(coords, target_var)
    # Per-location regularization losses
    reg_losses = dsntnn.js_reg_losses(heatmaps, target_var, sigma_t=1.0)
    # Combine losses into an overall loss
    loss = dsntnn.average_loss(euc_losses + reg_losses)
    #loss = mse(coords, target_var)
    losses += [loss.item()]
    euc += [euc_losses.mean().item()]
    reg += [reg_losses.mean().item()]
    # Calculate gradients
    optimizer.zero_grad()
    loss.backward()

    # Update model parameters with RMSprop
    optimizer.step()

plt.plot(losses)
plt.show()

plt.plot(reg)
plt.show()

plt.plot(euc)
plt.show()

# Predictions after training
print('Target coords: {:0.4f}, {:0.4f}'.format(*list(target_var[0, 0])))
print('Predicted coords: {:0.4f}, {:0.4f}'.format(*list(coords[0, 0])))

# for i in range(heatmaps.shape[0]):
#     plt.imshow(raccoon_face)
#     plt.imshow(heatmaps[i, 0].detach().cpu().numpy(), alpha=0.6)
#     plt.show()
    # plt.imshow(raccoon_face)
    # plt.imshow(heatmaps[i, 1].detach().cpu().numpy(), alpha=0.6)
    # plt.show()