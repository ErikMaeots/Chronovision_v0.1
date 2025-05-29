import torch
import json
from torch.utils.data import DataLoader
from model import DeepLabV3Custom_v3
from train import JointTransform, QuantificationDataset, NanometerLoss, JSDWithMask, HybridJSDLoss, Paper_figure_loss

checkpoint_path = '/mnt/g/fast_data_access_work/high_speed_camera_AI/code/checkpoints/model_v47/DeepLabV3Custom_Big_27.pth'
train_json_file_path = '/mnt/g/fast_data_access_work/high_speed_camera_AI/code/train_pairs_v3.json'
val_json_file_path = '/mnt/g/fast_data_access_work/high_speed_camera_AI/code/val_pairs_v3.json'

with open(train_json_file_path, 'r') as f:
    train_image_pairs = json.load(f)

with open(val_json_file_path, 'r') as f:
    val_image_pairs = json.load(f)

mean_input = torch.tensor([119.7732])
std_input = torch.tensor([66.1909])
mean_target = torch.tensor([203.6308])
std_target = torch.tensor([96.0295])

transform = JointTransform(mean_input, std_input, mean_target, std_target, p_flip=0.0, rotation_range=(0,0), scale_range=(1.0, 1.0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize datasets
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

# Create DataLoaders with batching
batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

# Initialize model and load checkpoint
# model = UNet(in_channels=1, out_channels=1).to(device)
model = DeepLabV3Custom_v3(in_channels=1, out_channels=1).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define the loss criterion
nanometer_loss = NanometerLoss(
    low=10,
    high=200,
    mean_target=mean_target.item(),
    std_target=std_target.item()
)
Paper_loss = Paper_figure_loss(
    min_value=10,
    max_value=200,
    mean_target=mean_target.item(),
    std_target=std_target.item())
distribution_loss = JSDWithMask(threshold=0.60, num_bins=50, sigma=0.005)  # 0.17 for 220nm
hybrid_loss = HybridJSDLoss()


def compute_losses_for_loader(loader):
    losses = {
        "nanometer_loss": [],
        "distribution_loss": [],
        "hybrid_loss": [],
        "focus": [],
        "Paper_loss": [],
    }

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs)

            for i in range(inputs.size(0)):
                # Calculate losses per sample
                sample_prediction = predictions[i:i + 1]
                sample_target = targets[i:i + 1]
                nanometer_loss_value = nanometer_loss(sample_prediction, sample_target).item()
                distribution_loss_value = distribution_loss(sample_prediction, sample_target).item()
                hybrid_loss_value = hybrid_loss(sample_prediction, sample_target).item()
                paper_loss_value = Paper_loss(sample_prediction, sample_target).item()

                # Append each loss to the corresponding list
                losses["nanometer_loss"].append(nanometer_loss_value)
                losses["distribution_loss"].append(distribution_loss_value)
                losses["hybrid_loss"].append(hybrid_loss_value)
                losses["Paper_loss"].append(paper_loss_value)
    return losses


train_losses = compute_losses_for_loader(train_loader)
val_losses = compute_losses_for_loader(val_loader)
#
# Save losses to json files
with open('train_losses_for_paper.json', 'w') as f:
    json.dump(train_losses, f, indent=4)

with open('val_losses_for_paper.json', 'w') as f:
    json.dump(val_losses, f, indent=4)