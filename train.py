import torch.nn as nn
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import random
from torchvision.transforms import v2, InterpolationMode
import json
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import os
from model import DeepLabV3Custom_v3
import math
from typing import Optional

'''
Mandatory preprocessing steps before training model
1. high-speed camera data is scaled by 10.79181 plus minus 2% depending on best fit to the EM data.
2. high-speed camera is further bilinearly scaled from 100x100 to 128x128 to have a nice input into the model
3. EM data is downsampled to 128x128 to have a nice target for the model.
4. since the original box sizes were chosen to match the scaling the final pixel size in both images is very similar,
but maybe a little inaccurate
5. Due to the intensity levels in each image I found that adding +70 to the high-speed camera data gives a good match of
vacuum intensity.
6. high-speed camera data is rotated and shifted for best fit.
'''


def rotate(image: torch.Tensor, angle: float, pad_mode: str = 'reflect') -> torch.Tensor:
    C, H, W = image.shape

    # Map pad_mode
    if pad_mode == 'nearest':
        pad_mode_torch = 'replicate'
    else:
        pad_mode_torch = 'reflect'

    # Compute image diagonal to ensure enough padding for any rotation
    diag = int(math.ceil(math.sqrt(H**2 + W**2)))
    # Pad to make the image at least diag x diag
    pad_h = max((diag - H) // 2, 0)
    pad_w = max((diag - W) // 2, 0)

    # Pad the image so that rotation won't sample outside
    # Padding order: left, right, top, bottom
    padded_image = F.pad(image, (pad_w, pad_w, pad_h, pad_h), padding_mode=pad_mode_torch)
    _, H_pad, W_pad = padded_image.shape

    # Rotate with expand=True so we get the full bounding box after rotation
    rotated = F.rotate(padded_image, angle, interpolation=InterpolationMode.BILINEAR,
                     expand=True, center=None, fill=None)
    _, H_rot, W_rot = rotated.shape
    delta_h = H - H_rot
    delta_w = W - W_rot

    if delta_h == 0 and delta_w == 0:
        # Perfect match
        return rotated

    if delta_h > 0 or delta_w > 0:
        # The rotated image is smaller than original (very unlikely with expand=True, but just in case)
        pad_top = delta_h // 2 if delta_h > 0 else 0
        pad_bottom = delta_h - pad_top if delta_h > 0 else 0
        pad_left = delta_w // 2 if delta_w > 0 else 0
        pad_right = delta_w - pad_left if delta_w > 0 else 0
        rotated = F.pad(rotated, (pad_left, pad_top, pad_right, pad_bottom), mode=pad_mode_torch)
    else:
        # The rotated image is larger, we need to crop
        extra_h = H_rot - H
        extra_w = W_rot - W
        crop_top = extra_h // 2
        crop_left = extra_w // 2
        rotated = rotated[:, crop_top:crop_top + H, crop_left:crop_left + W]

    return rotated


def resize(input_tensor, scale_factor, pad_mode='reflect'):
    C, H, W = input_tensor.shape
    x = input_tensor.unsqueeze(0)
    scaled = nn.functional.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    scaled = scaled.squeeze(0)  # (C, H_out, W_out)
    _, H_out, W_out = scaled.shape
    delta_h = H - H_out
    delta_w = W - W_out
    if pad_mode == 'nearest':
        pad_mode_torch = 'replicate'
    else:
        pad_mode_torch = pad_mode
    if delta_h == 0 and delta_w == 0:
        return scaled
    if delta_h > 0 or delta_w > 0:
        pad_top = delta_h // 2 if delta_h > 0 else 0
        pad_bottom = delta_h - pad_top if delta_h > 0 else 0
        pad_left = delta_w // 2 if delta_w > 0 else 0
        pad_right = delta_w - pad_left if delta_w > 0 else 0
        scaled = F.pad(scaled, (pad_left, pad_top, pad_right, pad_bottom), padding_mode=pad_mode_torch)
    else:
        extra_h = H_out - H
        extra_w = W_out - W
        crop_top = extra_h // 2 if extra_h > 0 else 0
        crop_left = extra_w // 2 if extra_w > 0 else 0
        scaled = scaled[:, crop_top:crop_top + H, crop_left:crop_left + W]
    return scaled


def create_weight_mask(num_bins, weight_middle=1.5, weight_ends=0.5):
    weights = torch.ones(num_bins)
    start_end = int(num_bins * 0.1)
    weights[:start_end] = weight_ends
    weights[-start_end:] = weight_ends
    weights[start_end:-start_end] = weight_middle
    return weights


class JSDWithMask(nn.Module):
    def __init__(self, threshold, num_bins=256, sigma=0.01, epsilon=1e-10):
        super(JSDWithMask, self).__init__()
        self.threshold = threshold
        self.num_bins = num_bins
        self.sigma = sigma
        self.epsilon = epsilon
        bin_edges = torch.linspace(-2.1, threshold, steps=num_bins, device='cuda')
        self.register_buffer('bins', bin_edges)
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.weight_mask = create_weight_mask(self.num_bins, weight_middle=1.0, weight_ends=0.25)

    def marginal_pdf(self, values):
        residuals = values.unsqueeze(-1) - self.bins.unsqueeze(0)
        kernel = torch.exp(-0.5 * (residuals / self.sigma) ** 2)
        pdf = torch.mean(kernel, dim=0)
        pdf = (pdf + self.epsilon) / (pdf.sum() + self.epsilon)
        return pdf

    # def display_marginal_pdf(self, pdf):
    #     if isinstance(pdf, torch.Tensor):
    #         pdf = pdf.cpu().numpy()  # Convert to NumPy array if it's a PyTorch tensor
    #     plt.figure(figsize=(8, 6))
    #     plt.bar(range(len(pdf)), pdf, width=1.0, align='center', alpha=0.75)
    #     plt.title("Marginal PDF")
    #     plt.xlabel("Bins")
    #     plt.ylabel("Probability")
    #     plt.grid(axis='y', linestyle='--', alpha=0.7)
    #     plt.show()

    def forward(self, p, q):
        N, C, H, W = p.shape
        p = p.view(N, C, -1)
        q = q.view(N, C, -1)
        p_masked  = p[p <= self.threshold]
        q_masked = q[q <= self.threshold]
        if p_masked.numel() == 0 or q_masked.numel() == 0:
            return torch.tensor(0.0, device=p.device)
        p_pdf = self.marginal_pdf(p_masked)
        q_pdf = self.marginal_pdf(q_masked)
        m = (0.5 * (p_pdf + q_pdf)).log()
        kl_p = self.kl(m, p_pdf.log())  # Shape: (batch_size, num_bins)
        kl_q = self.kl(m, q_pdf.log())
        weights = self.weight_mask.to(p.device)
        kl_p = kl_p * weights
        kl_q = kl_q * weights
        loss = 0.5 * (kl_p.sum(dim=-1).mean() + kl_q.sum(dim=-1).mean())
        return loss


class WeightedRangeMSELoss(nn.Module):
    def __init__(self, low: float, high: float,
                 in_range_weight: float = 1.0, out_of_range_weight: float = 0.5,
                 spatial_weight_mask: Optional[torch.Tensor] = None):
        super(WeightedRangeMSELoss, self).__init__()
        self.low = low
        self.high = high
        self.in_range_weight = in_range_weight
        self.out_of_range_weight = out_of_range_weight
        self.spatial_weight_mask = spatial_weight_mask

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_in_range = (target >= self.low) & (target <= self.high)
        value_based_weights = torch.where(
            target_in_range,
            torch.tensor(self.in_range_weight, device=pred.device),
            torch.tensor(self.out_of_range_weight, device=pred.device)
        )
        if self.spatial_weight_mask is not None:
            if self.spatial_weight_mask.size(0) != pred.size(0):
                spatial_mask = self.spatial_weight_mask[:pred.size(0)]
            else:
                spatial_mask = self.spatial_weight_mask
            weights = value_based_weights * spatial_mask
        else:
            weights = value_based_weights
        squared_errors = (pred - target) ** 2
        weighted_errors = squared_errors * weights
        return weighted_errors.sum() / weights.sum()


class HybridJSDLoss(nn.Module):
    def __init__(self):
        super(HybridJSDLoss, self).__init__()
        self.jsd = JSDWithMask(threshold=0.60, num_bins=50, sigma=0.005)
        self.mse = WeightedRangeMSELoss(low=-2, high=-0.038, spatial_weight_mask=None)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        jsd_loss = self.jsd(pred, target)
        mse_loss = self.mse(pred, target)
        return jsd_loss + mse_loss


class NanometerLoss(nn.Module):
    def __init__(self, low: float, high: float, mean_target: float, std_target: float,
                 binary_mask: Optional[torch.Tensor] = None):
        super(NanometerLoss, self).__init__()
        self.low = low
        self.high = high
        self.mean_target = mean_target
        self.std_target = std_target
        self.binary_mask = binary_mask

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.binary_mask is not None:
            if self.binary_mask.size(0) != pred.size(0):
                spatial_mask = self.binary_mask.repeat(pred.size(0), 1, 1, 1)
            else:
                spatial_mask = self.binary_mask
        else:
            device = pred.device
            spatial_mask = torch.ones_like(pred, dtype=torch.bool, device=device)
        pred_orig = pred * self.std_target + self.mean_target
        target_orig = target * self.std_target + self.mean_target
        target_in_range = (target_orig >= self.low) & (target_orig <= self.high)
        pred_in_range = (pred_orig >= self.low) & (pred_orig <= self.high)
        combined_mask = target_in_range | pred_in_range
        final_mask = combined_mask.bool() & spatial_mask.bool()
        if final_mask.sum() <= final_mask.numel() * 0.1:
            diff = torch.abs(pred_orig - target_orig) * spatial_mask
            return torch.mean(diff[diff != 0])
        focused_pred = pred_orig[final_mask]
        focused_target = target_orig[final_mask]
        diff = torch.abs(focused_pred - focused_target)
        return torch.mean(diff)


class Paper_figure_loss(nn.Module):
    def __init__(self, mean_target: float, std_target: float,
                 binary_mask: Optional[torch.Tensor] = None,
                 min_value: float = 10.0, max_value: float = 200.0):
        super(Paper_figure_loss, self).__init__()
        self.mean_target = mean_target
        self.std_target = std_target
        self.binary_mask = binary_mask
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.binary_mask is not None:
            if self.binary_mask.size(0) != pred.size(0):
                spatial_mask = self.binary_mask.repeat(pred.size(0), 1, 1, 1)
            else:
                spatial_mask = self.binary_mask
        else:
            spatial_mask = torch.ones_like(pred, dtype=torch.bool, device=pred.device)

        pred_orig = pred * self.std_target + self.mean_target
        target_orig = target * self.std_target + self.mean_target

        target_in_range = (target_orig >= self.min_value) & (target_orig <= self.max_value)

        final_mask = target_in_range & spatial_mask.bool()

        if final_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        focused_pred = pred_orig[final_mask]
        focused_target = target_orig[final_mask]

        diff = torch.abs(focused_pred - focused_target)
        return torch.mean(diff)


class JointTransform:
    def __init__(self, mean_input, std_input, mean_target, std_target, p_flip=0.5, rotation_range=(-180, 180), scale_range=(0.98, 1.02)):
        self.mean_input = mean_input
        self.std_input = std_input
        self.mean_target = mean_target
        self.std_target = std_target
        self.p_flip = p_flip
        self.rotation_range = rotation_range
        self.scale_range = scale_range

        # Deterministic transforms
        self.to_image = v2.ToImage()
        self.to_dtype = v2.ToDtype(torch.float32, scale=True)
        self.normalize_input = v2.Normalize(mean_input, std_input)
        self.normalize_target = v2.Normalize(mean_target, std_target)

    def __call__(self, input_image, target_image):
        # Convert images to tensors
        input_image = self.to_image(input_image)
        target_image = self.to_image(target_image)

        input_image = self.to_dtype(input_image)
        target_image = self.to_dtype(target_image)

        scale = random.uniform(*self.scale_range)
        input_image = resize(input_image, scale)
        target_image = resize(target_image, scale)

        angle = random.uniform(*self.rotation_range)
        input_image = rotate(input_image, angle, )
        target_image = rotate(target_image, angle)

        if torch.rand(1) < self.p_flip:
            input_image = F.hflip(input_image)
            target_image = F.hflip(target_image)
        if torch.rand(1) < self.p_flip:
            input_image = F.vflip(input_image)
            target_image = F.vflip(target_image)
        input_image = self.normalize_input(input_image)
        target_image = self.normalize_target(target_image)

        return input_image, target_image

class QuantificationDataset(Dataset):
    def __init__(self, input_dir, target_dir, input_images, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.input_images = input_images
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_images[idx]['input'])
        target_path = os.path.join(self.target_dir, self.input_images[idx]['target'])
        print(target_path, input_path)
        input_image = Image.open(input_path)
        target_image = Image.open(target_path)

        # Apply joint transforms
        if self.transform:
            input_image, target_image = self.transform(input_image, target_image)

        return input_image, target_image


if __name__ == "__main__":

    batch_size = 8
    num_workers = 8
    num_epochs = 100
    mean_input = torch.tensor([119.7732])
    std_input = torch.tensor([66.1909])
    mean_target = torch.tensor([203.6308])
    std_target = torch.tensor([96.0295])

    train_json_file_path = '/mnt/g/fast_data_access_work/high_speed_camera_AI/code/train_pairs_v3.json'
    val_json_file_path = '/mnt/g/fast_data_access_work/high_speed_camera_AI/code/val_pairs_v3.json'

    with open(train_json_file_path, 'r') as f:
        train_image_pairs = json.load(f)

    with open(val_json_file_path, 'r') as f:
        val_image_pairs = json.load(f)

    # Hyperparameters


    transform = JointTransform(mean_input, std_input, mean_target, std_target, p_flip=0.5)

    train_dataset = QuantificationDataset(
        input_dir='/mnt/g/fast_data_access_work/high_speed_camera_AI/Dataset/preprocessed_inputs_fixed/',
        target_dir='/mnt/g/fast_data_access_work/high_speed_camera_AI/Dataset/preprocessed_targets/',
        input_images=train_image_pairs,
        transform=transform
    )

    val_dataset = QuantificationDataset(
        input_dir='/mnt/g/fast_data_access_work/high_speed_camera_AI/Dataset/preprocessed_inputs_fixed/',
        target_dir='/mnt/g/fast_data_access_work/high_speed_camera_AI/Dataset/preprocessed_targets/',
        input_images=val_image_pairs,
        transform=transform
    )

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    learning_rate = 5E-07
    log_dir = f'./logs/model_v47'
    save_dir = f'./checkpoints/model_v47'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model = DeepLabV3Custom_v3(in_channels=1, out_channels=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_loader), epochs=50)
    binary_mask = torch.load('disk_mask_binary.pt').to(device)
    criterion = HybridJSDLoss()
    nanometer_loss = NanometerLoss(
        low=10,
        high=200,
        mean_target=mean_target.item(),
        std_target=std_target.item(),
        binary_mask=binary_mask[0, :, :, :],
    )

    checkpoint_path = "checkpoints/model_v46/DeepLabV3Custom_Big_13.pth"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    except FileNotFoundError:
        print("No checkpoint found. Starting training from scratch.")
        exit(0)