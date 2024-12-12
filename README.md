# Project Name
**CS445 Fall 2024 Final Project** University of Illinois  

<ins>Group members:</ins>  
* Matt Poteshman (**mrp12**)  
* Paul Lambert (**lamber10**)  

## Background
Project information and ideas. Texture transfer, but from talented artists! 

<p align="center">
  <img src="docs/images/b0.png" width="45%" />
  <img src="docs/images/b1.png" width="45%" />
</p>

*NST applied to a photograph of Maui's Waiʻānapanapa State Park, with style transfer from Van Gogh's* Starry Night


## Instructions
To build and run the project, first initialize your environment using your preferrred Python package manager:

#### Conda:
```bash
conda env create -f environment.yml
conda activate finalproj
```

The `environment.yml` already includes pip dependencies, so no additional pip install is needed. 

#### PIP:
```bash
python -m venv .myvenv
source .myvenv/bin/activate
pip install -r requirements.txt
```

Note: CUDA version 12.4 is required for this environment. If using PIP, make sure the GPU-accelerated versions of PyTorch are installed.


### Neural Style Transfer
```bash
python src/nst.py
```

Usage:
* `--input='<input_path.jpg>'` image to be altered
* `--style='<style_path.jpg>'` style source image (default: 'images/art/starry_night.jpg')
* `--gamma` color preservation weight on loss function (optional, default `1e5`)
* `--color_control` color content preservation (optional, default `0.7`)


### CycleGAN
**To train:**  
```
python src/cyclegan.py --train /
                       --style_dir "images/art/vangogh/" /
                       --epochs 100 /
                       --batch_size 1 /
                       --lr 0.0002
```    

**Inference:**  
```bash
python src/cyclegan.py --input_image "images/input/dummy_class/input.jpg" /
                       --output_image "images/output/styled_vangogh.jpg"
```


## Sources

### Art Sources

[Van Gogh's works](https://www.nga.gov/collection/artist-info.1349.html#works)

#### Currently not used
* https://drive.google.com/drive/folders/1CglMyDFXJFNpDt3ebstOPYVwnVvMNv6g
* https://www.reddit.com/r/DataHoarder/comments/d0wuae/50k_images_from_the_art_institute_of_chicago/
* https://www.reddit.com/r/TheFrame/comments/10cu8hg/over_400_4k_opensource_artworks_from_around_the/
