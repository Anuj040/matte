# Real-Time High-Resolution Background Matting
This is a reimplementation of the work by the same name, [Real-Time High-Resolution Background Matting](https://arxiv.org/abs/2012.07810). The work borrows a significant bit of original [code](https://github.com/PeterL1n/BackgroundMattingV2).
At the same time, there are several changes, reflective of my own coding style.
Please feel free to use, fork the code or leave comments if you find any inconsitency.

## Environment
* torch == 1.9.0
* numpy == 1.20.3
* kornia == 0.4.1
* torchvision == 0.10.0

## Installation Guideline

```sh
git clone git@github.com:Anuj040/matte.git [-b <branch_name>]
cd matte (Work Directory)

# local environment settings
pyenv local 3.8.10                                 
python -m pip install poetry
poetry config virtualenvs.create true --local
poetry config virtualenvs.in-project true --local

# In case older version of pip throws installation errors
poetry run python -m pip install --upgrade pip 

# local environment preparation
poetry install

```
## Working with the code
Before running the code please make sure that your datasets follow the below directory hierarchy.

```sh
    matte {work_directory}
    ├── src
    ├── datasets                  
    │   ├──PhotoMatte85                                   
    │       ├──train                               # RGBA image files 
    │         ├──fgr+alpha_1.png
    │         ├──fgr+alpha_1.png
    │       ├──valid
    │         :
    │   ├──backgrounds
    │       ├──train                               
    │         ├──bg_image1.png                      # RGB image file
    │         ├──bg_image2.png
    │           :  
    │       ├──valid                                
    │         ├──bg_image3.png
    │         ├──bg_image4.png
    │           :      
    └── ...
```
## Notes
1. Inputs to the model are foreground (RGBA-4 channel) and background images
2. Refer _data_path.py_ to get a better idea of how to specify dataset paths 

### Training mode
1. All the code executions shall happen from work directory
2. There are three different model train modes (base, refine, gan). The former two are as in the original implementation. The last one is _refine_ mode with additional discriminator assitance.
3. The code also includes functionality for a two step training (training base model first, and then finetuning it with training of refiner model). It can be assessed with _--load_base_ flag. 
However, the path to the pretrained _base model_ is hard coded as of now. Please change it as per your convenience.
4. In my limited set of experiments following the 2-step training strategy as described above leads to faster model convergence for refine model.

  ```sh
  poetry run python src/run.py train --model_type=refine
  ```
