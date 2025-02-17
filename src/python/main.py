import pdb
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define data transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the training dataset
train_dataset = datasets.ImageNet(root='/Users/mbenisch/data/imagenet', split='train', transform=transform)

# Create a data loader for the training dataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

# Load the validation dataset
val_dataset = datasets.ImageNet(root='/Users/mbenisch/data/imagenet', split='val', transform=transform)

# Create a data loader for the validation dataset
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

pdb.set_trace()
