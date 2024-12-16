import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CycleGANDataset(Dataset):
    def __init__(self, root_A, root_B, transform=None):
        self.root_A = root_A
        self.root_B = root_B
        
        # Get all image files
        self.files_A = sorted([f for f in os.listdir(root_A) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.files_B = sorted([f for f in os.listdir(root_B) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(512),
                transforms.RandomCrop(512),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, idx):
        # handle if one domain has fewer images than the other
        idx_A = idx % len(self.files_A)
        idx_B = idx % len(self.files_B)
        
        # load images
        img_A = Image.open(os.path.join(self.root_A, self.files_A[idx_A])).convert('RGB')
        img_B = Image.open(os.path.join(self.root_B, self.files_B[idx_B])).convert('RGB')
        
        # apply transforms
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        
        return {'A': img_A, 'B': img_B}
