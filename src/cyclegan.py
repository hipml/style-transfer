import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
import time
from pathlib import Path
import itertools  
import argparse

from cg_generator import StrokeAwareGenerator
from cg_discriminator import Discriminator
from cg_dataset import CycleGANDataset


class CycleGANTrainer:
    def __init__(self, generator_ab, generator_ba, discriminator_a,
                 discriminator_b, device='cuda' if torch.cuda.is_available() else 'cpu',
                 image_size=256):
        
        self.G_AB = generator_ab.to(device)
        self.G_BA = generator_ba.to(device)
        self.D_A = discriminator_a.to(device)
        self.D_B = discriminator_b.to(device)
        self.device = device
        self.image_size = image_size
        
        self.criterion_gan = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()),
            lr=0.0001, betas=(0.5, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
            lr=0.0001, betas=(0.5, 0.999)
        )
        
        self.lambda_cycle = 10.0
        self.lambda_identity = 0.5
        
        self.setup_directories()
        
    def setup_directories(self):
        self.save_dir = Path('outputs')
        self.image_dir = self.save_dir / 'images'
        self.model_dir = self.save_dir / 'models'
        
        self.save_dir.mkdir(exist_ok=True)
        self.image_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)

    def set_input(self, batch):
        self.real_A = batch['A'].to(self.device)
        self.real_B = batch['B'].to(self.device)
        torch.cuda.empty_cache()

    def train_discriminators(self):
        self.optimizer_D.zero_grad()
        
        with torch.no_grad():
            fake_B = self.G_AB(self.real_A)
            fake_A = self.G_BA(self.real_B)
        
        with torch.amp.autocast(device_type='cuda'):
            loss_D_A_real = self.criterion_gan(self.D_A(self.real_A), torch.ones_like(self.D_A(self.real_A)))
            loss_D_B_real = self.criterion_gan(self.D_B(self.real_B), torch.ones_like(self.D_B(self.real_B)))

            loss_D_A_fake = self.criterion_gan(self.D_A(fake_A),  torch.zeros_like(self.D_A(fake_A)))
            loss_D_B_fake = self.criterion_gan(self.D_B(fake_B), torch.zeros_like(self.D_B(fake_B)))
        
        loss_D = (loss_D_A_real + loss_D_A_fake) * 0.5 + (loss_D_B_real + loss_D_B_fake) * 0.5
        loss_D.backward()
        self.optimizer_D.step()
        
        torch.cuda.empty_cache()
        return loss_D.item()

    def train_generators(self):
        self.optimizer_G.zero_grad()
        
        with torch.amp.autocast(device_type='cuda'):
            identity_A = self.G_BA(self.real_A)
            identity_B = self.G_AB(self.real_B)
            loss_identity = (self.criterion_identity(identity_A, self.real_A) +
                           self.criterion_identity(identity_B, self.real_B)) * self.lambda_identity
            
            fake_B = self.G_AB(self.real_A)
            loss_G_AB = self.criterion_gan(self.D_B(fake_B), torch.ones_like(self.D_B(fake_B)))
            fake_A = self.G_BA(self.real_B)
            loss_G_BA = self.criterion_gan(self.D_A(fake_A), torch.ones_like(self.D_A(fake_A)))
            
            cycle_A = self.G_BA(fake_B)
            cycle_B = self.G_AB(fake_A)
            loss_cycle = (self.criterion_cycle(cycle_A, self.real_A) +
                         self.criterion_cycle(cycle_B, self.real_B)) * self.lambda_cycle
        
        loss_G = loss_G_AB + loss_G_BA + loss_cycle + loss_identity
        loss_G.backward()
        self.optimizer_G.step()
        
        self.fake_A = fake_A.detach()
        self.fake_B = fake_B.detach()
        self.cycle_A = cycle_A.detach()
        self.cycle_B = cycle_B.detach()
        
        torch.cuda.empty_cache()
        return loss_G.item()

    def save_images(self, epoch, batch_idx):
        images = {
            'real_A': self.real_A,
            'real_B': self.real_B,
            'fake_A': self.fake_A,
            'fake_B': self.fake_B,
            'cycle_A': self.cycle_A,
            'cycle_B': self.cycle_B
        }
        
        for name, img in images.items():
            save_image(img, self.image_dir / f'{name}_epoch{epoch}_batch{batch_idx}.png', normalize=True)

    def save_models(self, epoch):
        torch.save({
            'G_AB': self.G_AB.state_dict(),
            'G_BA': self.G_BA.state_dict(),
            'D_A': self.D_A.state_dict(),
            'D_B': self.D_B.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'epoch': epoch
        }, self.model_dir / f'checkpoint_epoch{epoch}.pth')

    def load_models(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.G_AB.load_state_dict(checkpoint['G_AB'])
        self.G_BA.load_state_dict(checkpoint['G_BA'])
        self.D_A.load_state_dict(checkpoint['D_A'])
        self.D_B.load_state_dict(checkpoint['D_B'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        return checkpoint['epoch']

    def train(self, dataloader, num_epochs, save_frequency=10):
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            for i, batch in enumerate(dataloader):
                try:
                    self.set_input(batch)
                    loss_D = self.train_discriminators()
                    loss_G = self.train_generators()
                    
                    if i %10 == 0:
                        print(f'Epoch {epoch}/{num_epochs} [{i}/{len(dataloader)}] '
                              f'Loss_D: {loss_D:.4f} Loss_G: {loss_G:.4f}')
                        self.save_images(epoch, i)
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print('Out of memory! Clearing cache and skipping batch...')
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
            if (epoch + 1) % save_frequency == 0:
                self.save_models(epoch)
            
            epoch_time = time.time() - epoch_start_time
            print(f'End of epoch {epoch}. Time taken: {epoch_time:.2f}s')
        
        total_time = time.time() - start_time
        print(f'Training finished! Total time: {total_time/3600:.2f} hours')

    def load_image(self, image_path):
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(self.device)
        return image

    def inference(self, image_path, direction='AtoB', output_path=None):
        self.G_AB.eval()
        self.G_BA.eval()
        
        with torch.no_grad():
            # Load and preprocess image
            input_image = self.load_image(image_path)
            
            # Generate output
            if direction == 'AtoB':
                output = self.G_AB(input_image)
            else:  # BtoA
                output = self.G_BA(input_image)
            
            # Save result
            if output_path is None:
                output_path = self.image_dir / f'inference_{Path(image_path).stem}_{direction}.png'
            
            save_image(output, output_path, normalize=True)
            
            return output

    def batch_inference(self, input_dir, direction='AtoB', output_dir=None):
        if output_dir is None:
            output_dir = self.image_dir / 'inference'
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        input_dir = Path(input_dir)
        image_files = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png'))
        
        for image_file in image_files:
            try:
                output_path = output_dir / f'{image_file.stem}_{direction}{image_file.suffix}'
                self.inference(image_file, direction, output_path)
                print(f'Processed {image_file.name}')
            except Exception as e:
                print(f'Error processing {image_file.name}: {str(e)}')

def train(args):
    G_AB = StrokeAwareGenerator()
    G_BA = StrokeAwareGenerator()
    D_A = Discriminator()
    D_B = Discriminator()
    
    dataset = CycleGANDataset(args.input_dir, args.style_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    trainer = CycleGANTrainer(G_AB, G_BA, D_A, D_B)
    
    if args.resume and args.checkpoint:
        start_epoch = trainer.load_models(args.checkpoint)
        print(f'Resuming from epoch {start_epoch}')
    
    trainer.train(dataloader, num_epochs=args.epochs, save_frequency=args.save_freq)

def generate(args):
    G_AB = StrokeAwareGenerator()
    G_BA = StrokeAwareGenerator()
    D_A = Discriminator()
    D_B = Discriminator()
    
    trainer = CycleGANTrainer(G_AB, G_BA, D_A, D_B)
    
    if not args.checkpoint:
        raise ValueError("Checkpoint path is required for generation mode")
    
    trainer.load_models(args.checkpoint)
    print(f'Loaded checkpoint: {args.checkpoint}')
    
    if os.path.isfile(args.input):
        trainer.inference(
            args.input, 
            direction=args.direction,
            output_path=args.output
        )
        print(f'Generated output saved to: {args.output}')
    else:
        trainer.batch_inference(
            args.input,
            direction=args.direction,
            output_dir=args.output
        )
        print(f'Generated outputs saved to directory: {args.output}')

def main():
    parser = argparse.ArgumentParser(description='CycleGAN Training and Generation')
    
    parser.add_argument('--checkpoint', default='outputs/models/checkpoint_epoch199.pth', type=str, help='Path to checkpoint file')
    
    subparsers = parser.add_subparsers(dest='mode', help='Mode')
    
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--input_dir', required=True, help='Input images')
    train_parser.add_argument('--style_dir', required=True, help='Style images')
    train_parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    train_parser.add_argument('--save-freq', type=int, default=10, help='Checkpoint save frequency')
    train_parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    
    gen_parser = subparsers.add_parser('generate', help='Generate styled images')
    gen_parser.add_argument('--input', required=True, help='Input image or directory')
    gen_parser.add_argument('--output', required=True, help='Output path or directory')
    gen_parser.add_argument('--direction', choices=['AtoB', 'BtoA'], default='AtoB',
                           help='Translation direction')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'generate':
        generate(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
