
This repository contains the 3rd place solution for the Kaggle competition "Generative Dog Images" https://www.kaggle.com/c/generative-dog-images/overview.

# Directories
kaggle-gendog/code: the source code including the submitted kernel script SAGAN-private-66.ipynb to run in Kaggle kernel and modules to run in command line.
kaggle-gendog/sagan_models: contains the generated generator model that can be used to generate samples directly.
kaggle-gendog/sample_images: contains a few sample images generated from the model.

# SOFTWARE
The solution is created and tested on Kaggle kernels with Python 3.6.6.
The rest of software packages is specified in requirements.txt

# DATA 
The training data can be downloaded from Kaggle website https://www.kaggle.com/c/generative-dog-images/data and put in the folder ./input/

# Shell commands to train and generate submissions 
## 1) generate submissions with the existing generator model
go to the code folder kaggle-gendog/code and run the script generate-samples.py

$cd kaggle-gendog/code 
$python generate_samples.py 

This command will use the model in the folder kaggle-gendog/sagan_models to generate 10000 sample images and the zipped image file as the submissions.
The generated images and submission are stored in the folder kaggle-gendog/output_images and kaggle-gendog/submissions.

if you wish to use a generator model in different folder and store the generated images in different folder, just specify your folders in the commandline such as,

$python generate_samples.py --save_model_dir "your model dir" --sample_images_path "your output images dir" --submission_dir --num_sample_images 100

## 2) retrain the model
go to the code folder kaggle-gendog/code and run the script train.py

$cd kaggle-gendog/code 
$python train.py

This command will use the data in the folder 'kaggle-gendog/input' to train the model and save the generator model in the folder kaggle-gendog/sagan_models.

If you wish to use a different data folder or save the model to a different folder etc, you can run the script with options listed below, 

--dataroot: specify the train data root folder, default: kaggle-gendog/input
--save_model_dir: specify the folder to store the generator model, default: kaggle-gendog/sagan_models
--num_epoches: specify the number of training epoch, default: 170


All the options and their default values for reproducing the result are specified in kaggle-gendog/code/parameters.py

Note: the train data folder must be structured as below,
dataroot/all-dogs/all-dogs/: all image files
dataroot/annotation/Annotation/: annotations stored in each corresponding breed directory


# Jupiter notebook to run in Kaggle kernels 
The easiest way to reproduce the result is to fork and run the Kaggle kernel https://www.kaggle.com/lisali/sagan-submit-2?scriptVersionId=18714508
or 
https://www.kaggle.com/lisali/sagan-submit-2-b13ab5 which generates the submission and the generator model as well.
