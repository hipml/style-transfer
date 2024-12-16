import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import LBFGS 

import torchvision.models as models
from torchvision.models import VGG19_Weights
import torchvision.transforms as transforms
from torchvision.utils import save_image

import cv2
import numpy as np

import argparse
from tqdm import tqdm
from pathlib import Path
from PIL import Image

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        # let's get the convolutional layers after MaxPool layers
<<<<<<< HEAD
        self.select_features = ['0', '5', '10', '19', '28']
        self.vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features


    def forward(self, output):
        features = []
        for name, layer in self.vgg._modules.items():
            output = layer(output)
            if name in self.select_features:
                features.append(output)
=======
        self.layers = ['0', '5', '10', '19', '28']
        self.vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features

    def forward(self, output):
        features = []

        for name, layer in self.vgg._modules.items():
            # send through the layer
            output = layer(output)
            if name in self.layers:
                features.append(output)

>>>>>>> dev
        return features

def tensor_to_cv2(tensor):
    """Convert normalized PyTorch tensor to cv2 BGR format"""

    # denormalize
    denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
    rgb = denorm(tensor.squeeze()).clamp(0, 1)
    
    # convert to numpy and correct format
    np_img = (rgb.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    # convert RGB to BGR
    return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

def cv2_to_tensor(img, device):
    """Convert cv2 BGR image back to normalized PyTorch tensor"""

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0

    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).to(device)
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    return norm(tensor).unsqueeze(0)

def get_color_preservation_loss(target, content, color_weight):
    """
    Calculate color preservation loss using LAB color space with cv2
    """
<<<<<<< HEAD
=======

>>>>>>> dev
    target_bgr = tensor_to_cv2(target)
    content_bgr = tensor_to_cv2(content)
    
    # convert to LAB
    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    content_lab = cv2.cvtColor(content_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # normalize LAB values, L to [0, 1] and A,B to [-1, 1]
    target_lab[:,:,0] = target_lab[:,:,0] / 100.0
    content_lab[:,:,0] = content_lab[:,:,0] / 100.0
    target_lab[:,:,1:] = target_lab[:,:,1:] / 127.0
    content_lab[:,:,1:] = content_lab[:,:,1:] / 127.0
    
    # only compare a and b channels
    target_ab = torch.from_numpy(target_lab[:,:,1:]).to(target.device)
    content_ab = torch.from_numpy(content_lab[:,:,1:]).to(content.device)

    # color_diff = torch.abs(target_ab - content_ab)
    # loss = color_weight * torch.mean(color_diff ** 4)

    loss = color_weight * F.mse_loss(target_ab, content_ab)
    return loss

# content loss function
def get_content_loss(target, content):
    """
    Calculates the loss between content image and target image
    
    L_content(p, x, l) = 1/2 * (sum (F_ij - P_ij, ij)^2)
    """
<<<<<<< HEAD
=======

>>>>>>> dev
    return (0.5 * torch.mean((target - content)**2))

def get_style_loss(target, style):
    """
    style loss = equivalent to computing the maximum mean discrepancy between two images
<<<<<<< HEAD
    """ 
=======
    """

>>>>>>> dev
    G = gram_matrix(target)
    S = gram_matrix(style)
    return torch.mean((G - S)**2)

def gram_matrix(input):
    """
<<<<<<< HEAD
    Compute the Gram matrix for each layer
=======
>>>>>>> dev
    gram matrix is calculated for every layer

    G_ij = sum(F_ik * F_jk, k)
    """

    batch_size, channels, height, width = input.size()
    features = input.view(batch_size * channels, height * width)
    gram = torch.mm(features, features.t())
    return gram.div(channels * height * width)

def load_img(path, loader):
    img = Image.open(path)
    img = loader(img).unsqueeze(0)
    return img

<<<<<<< HEAD
def save(target, i):
    output_dir = Path('output')
=======
def save(target, i, output_dir, output_name):
    output_dir = Path(output_dir)
>>>>>>> dev
    output_dir.mkdir(exist_ok=True)
    
    denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
    img = target.clone().squeeze()
    img = denorm(img).clamp(0, 1)
<<<<<<< HEAD
    save_image(img, output_dir / f'result_{i}.png')
=======
    save_image(img, output_dir / f'{output_name}_{i}.png')
>>>>>>> dev

def parse_arguments():
    parser = argparse.ArgumentParser(description='Style Transfer Parameters')
    parser.add_argument('--input', default='images/input/hawaii.JPG', help='Path to input image')
    parser.add_argument('--style', default='images/art/starry_night.jpg', help='Path to style image')
<<<<<<< HEAD
    parser.add_argument('--gamma', type=float, default=1e5, help='Gamma parameter (default: 1e5)')
    parser.add_argument('--color_control', type=float, default=0.7, help='Color control parameter (default: 0.7)')
=======
    parser.add_argument('--alpha', type=float, default=1, help='Alpha parameter (default: 1)')
    parser.add_argument('--beta', type=float, default=1e7, help='Beta parameter (default: 1e7)')
    parser.add_argument('--gamma', type=float, default=1e3, help='Gamma parameter (default: 1e3)')
    parser.add_argument('--color_control', type=float, default=0.7, help='Color control parameter (default: 0.7)')
    parser.add_argument('--output_dir', default='output/', help='Output dir')
    parser.add_argument('--output_name', default='result.jpg', help='Output file name')
>>>>>>> dev
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vgg = VGG().to(device).eval()

        img_size = 512 if torch.cuda.is_available() else 128

        # normalizing using imagenet mean and stddev values for RGB
        loader = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

<<<<<<< HEAD
        steps = 1000
        alpha = 1 # content weight
        beta = 1e7 #style weight
=======
        steps = 500
        alpha = args.alpha # content weight
        beta = args.beta #style weight
>>>>>>> dev
        gamma = args.gamma # color preservation weight
        color_control = args.color_control # how much to preserve content colors, 0-1

        content_img = load_img(args.input, loader).to(device)
        style_img = load_img(args.style, loader).to(device)

        # initialize output to be a random noise image
        # target_img = torch.randn_like(content_img).to(device).requires_grad_(True)
        # optimizer = optim.Adam([target_img], lr=1e-3)

        # LBFGS is used in the original paper
        target_img = content_img.clone().requires_grad_(True)
        optimizer = LBFGS([target_img], max_iter=1, line_search_fn='strong_wolfe')
        iteration = [0]
        def closure():
<<<<<<< HEAD
=======

            # LBFGS can run for quite a bit, especially on some of our
            # combinations of inputs/style images. We will limit it to
            # 500 optimizer steps, which is low, but is near convergence
            # for most of our data set
            if iteration[0] > steps:
                return
            
>>>>>>> dev
            optimizer.zero_grad()

            # get features
            target_features = vgg(target_img)
            content_features = vgg(content_img)
            style_features = vgg(style_img)

            # calculate losses
            style_loss = 0
            content_loss = 0

            for target, content, style in zip(target_features, content_features, style_features):
                content_loss += get_content_loss(target, content)
                style_loss += get_style_loss(target, style)

            color_loss = get_color_preservation_loss(target_img, content_img, color_control)

            # calculate total loss according to paper
            # adding a color retaining weight
<<<<<<< HEAD
            total_loss = alpha * content_loss + beta * style_loss # + gamma * color_loss
=======
            total_loss = alpha * content_loss + beta * style_loss + gamma * color_loss
>>>>>>> dev

            # set parameters to zero, compute gradient, update parameters
            total_loss.backward()

            if iteration[0] % 50 == 0:
                print(f'step: {iteration[0]}, content loss: {content_loss.item()}, style loss: {style_loss.item()}')
<<<<<<< HEAD
            if iteration[0] %100 == 0:
                save(target_img, iteration[0])
=======
            if iteration[0] %500 == 0 and iteration[0] > 0:
                save(target_img, iteration[0], args.output_dir, args.output_name)
>>>>>>> dev

            iteration[0] += 1

            return total_loss

        for _ in range(steps):
            optimizer.step(closure)
        
        save(target_img, "final")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        torch.cuda.empty_cache()
    
if __name__ == "__main__":
    main()
