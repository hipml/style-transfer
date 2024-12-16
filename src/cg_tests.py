import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from cg_generator import StrokeAwareGenerator
from cg_discriminator import Discriminator
from cg_dataset import CycleGANDataset

def test_generator():
    # 1. basic shape test
    generator = StrokeAwareGenerator()
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 512, 512)
    output = generator(test_input)
    
    # check output shape
    assert output.shape == (batch_size, 3, 512, 512), f"Expected shape {(batch_size, 3, 512, 512)}, got {output.shape}"
    
    # 2. check output range (should be between -1 and 1 due to tanh)
    assert torch.min(output).item() >= -1.0, f"Min value {torch.min(output).item()} is less than -1"
    assert torch.max(output).item() <= 1.0, f"Max value {torch.max(output).item()} is greater than 1"
    
    # 3. test with a real image
    def load_test_image(path, size=512):
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img = Image.open(path).convert('RGB')
        return transform(img).unsqueeze(0)  # Add batch dimension
    
    def save_output_image(tensor, path):
        image = tensor.squeeze(0).cpu().detach().numpy()
        image = (image + 1) / 2.0  
        image = np.transpose(image, (1, 2, 0))  
        image = (image * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(image).save(path)
    
    # 4. test gradient flow
    test_input.requires_grad_(True)
    output = generator(test_input)
    loss = output.mean()
    loss.backward()
    
    assert test_input.grad is not None, "Gradients are not flowing properly"

    test_img = load_test_image('images/input/hawaii.JPG')
    with torch.no_grad():
        output = generator(test_img)
    save_output_image(output, 'output.jpg')
    
    print("All tests passed!")
    return generator

def test_discriminator():
    discriminator = Discriminator()
    
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 512, 512, requires_grad=True)
    
    output = discriminator(test_input)
    
    print(f"Discriminator output shape: {output.shape}")
    
    output.mean().backward()
    assert test_input.grad is not None, "Gradients aren't flowing properly"
    print("Gradient check passed!")
    
    assert len(output.shape) == 4, "Output should be a 4D tensor (batch_size, 1, H, W)"
    
    print("All discriminator tests passed!")
    return discriminator


def test_dataset():
    import numpy as np
    import tempfile
    import shutil
    
    # create temp dirs
    temp_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(temp_dir, 'A'))
    os.makedirs(os.path.join(temp_dir, 'B'))
    
    # create sample images
    for domain in ['A', 'B']:
        for i in range(2):
            img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            img.save(os.path.join(temp_dir, domain, f'img_{i}.jpg'))
    
    dataset = CycleGANDataset(
        root_A=os.path.join(temp_dir, 'A'),
        root_B=os.path.join(temp_dir, 'B')
    )
    
    sample = dataset[0]
    assert isinstance(sample, dict), "Sample should be a dictionary"
    assert 'A' in sample and 'B' in sample, "Sample should contain both domains"
    assert sample['A'].shape == (3, 512, 512), f"Unexpected shape: {sample['A'].shape}"
    
    # clean up
    shutil.rmtree(temp_dir)
    print("Dataset tests passed!")
    return dataset

def visualize_dataset_samples(dataset, num_samples=3):
    """Visualize samples from both domains"""
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
    
    for i in range(num_samples):
        sample = dataset[i]
        
        # denormalize from [-1,1] to [0,1]
        img_A = (sample['A'] + 1) / 2.0
        img_B = (sample['B'] + 1) / 2.0
        
        # move channels to end and convert to numpy for matplotlib
        img_A = img_A.permute(1, 2, 0).numpy().clip(0, 1)
        img_B = img_B.permute(1, 2, 0).numpy().clip(0, 1)
        
        # display images
        axes[0, i].imshow(img_A)
        axes[1, i].imshow(img_B)
        axes[0, i].axis('off')
        axes[1, i].axis('off')
        
    axes[0, 0].set_title('Domain A (Source)', pad=20)
    axes[1, 0].set_title('Domain B (Style)', pad=20)
    
    plt.tight_layout()
    plt.show()

def test_real_dataset():
    source_dir = "images/input"
    style_dir = "images/art/vangogh"
    
    try:
        dataset = CycleGANDataset(source_dir, style_dir)
        print(f"Successfully loaded dataset with {len(dataset)} pairs")
        print(f"Found {len(dataset.files_A)} images in source domain")
        print(f"Found {len(dataset.files_B)} images in style domain")
        
        # Visualize some samples
        visualize_dataset_samples(dataset)
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")

if __name__ == "__main__":
    generator = test_generator()
    discriminator = test_discriminator()
    test_dataset()
    test_real_dataset()
