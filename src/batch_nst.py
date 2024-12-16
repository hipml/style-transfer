import argparse
import os
import subprocess
from itertools import product
from pathlib import Path

def get_image_files(directory):
    image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    return [f for f in os.listdir(directory) if f.lower().endswith(image_extensions)]

def process_images(input_dir, style_dir, alpha=1, beta=1e7, gamma=1e3, color_control=0.7):
    input_images = get_image_files(input_dir)
    style_images = get_image_files(style_dir)
    
    output_base = Path('output')
    output_base.mkdir(exist_ok=True)
    
    total_combinations = len(input_images) * len(style_images)
    processed = 0
    
    for input_img, style_img in product(input_images, style_images):
        processed += 1
        print(f"\nProcessing combination {processed}/{total_combinations}")
        print(f"Input: {input_img}")
        print(f"Style: {style_img}")
        
        style_name = Path(style_img).stem
        output_dir = output_base / style_name
        output_dir.mkdir(exist_ok=True)
        
        cmd = [
            'python', 'src/nst.py',
            '--input', str(Path(input_dir) / input_img),
            '--style', str(Path(style_dir) / style_img),
            '--alpha', str(alpha),
            '--beta', str(beta),
            '--gamma', str(gamma),
            '--color_control', str(color_control),
            '--output_dir', str(output_dir),
            '--output_name', f'{input_img}'
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully processed {input_img} with style {style_img}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {input_img} with style {style_img}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch Neural Style Transfer Processing')
    parser.add_argument('--input_dir', default='images/input', help='input images')
    parser.add_argument('--style_dir', default='images/style', help='style images')
    parser.add_argument('--alpha', type=float, default=1, help='Alpha (default: 1)')
    parser.add_argument('--beta', type=float, default=1e7, help='Beta (default: 1e7)')
    parser.add_argument('--gamma', type=float, default=1e3, help='Gamma (default: 1e3)')
    parser.add_argument('--color_control', type=float, default=0.7, help='Color control (default: 0.7)')
    
    args = parser.parse_args()
    
    process_images(
        args.input_dir,
        args.style_dir,
        args.alpha,
        args.beta,
        args.gamma,
        args.color_control
    )
