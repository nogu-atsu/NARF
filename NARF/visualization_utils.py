import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity


def save_img(batch, name):  # b x 3 x size x size
    if isinstance(batch, torch.Tensor):
        batch = batch.data.cpu().numpy()
    if len(batch.shape) == 3:
        batch = np.tile(batch[:, None], (1, 3, 1, 1))
    b, _, size, _ = batch.shape
    n = int(b ** 0.5)

    batch = batch.transpose(0, 2, 3, 1)
    batch = batch[:n ** 2].reshape(n, n, size, size, 3)
    batch = np.concatenate(batch, axis=1)
    batch = np.concatenate(batch, axis=1)
    batch = np.clip(batch * 127.5 + 127.5, 0, 255).astype("uint8")
    batch = Image.fromarray(batch)
    batch.save(name)


def ssim(img1, img2):
    img1 = img1[0].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5  # size x size x 3
    img2 = img2[0].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5  # size x size x 3
    return structural_similarity(img1, img2, data_range=1, multichannel=True)  # scalar


def psnr(img1, img2):
    mse = F.mse_loss(img1, img2).item()
    return 20 * np.log10(2) - 10 * np.log10(mse)
