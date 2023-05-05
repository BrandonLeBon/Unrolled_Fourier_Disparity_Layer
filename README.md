# Unrolled Fourier Disparity Layers

This repository is the pytorch implementation of the method proposed in the paper: [*B. Le Bon, M. Le Pendu, C. Guillemot. "Unrolled Fourier Disparity Layer Optimization For Scene Reconstruction From Few-Shots Focal Stacks", ICASSP 2023*](https://hal.science/hal-04054360/).

# Usage

The purpose of this code is to unroll the Fourier Disparity Layers optimization in order to reconstruct a light field or a set of refocus images from focal stack images as measurements. 

## Training

### Preparation

Before launching the training, you need to prepare the following files:
* Training dataset and validation dataset files listing the path to the corresponding dataset folders. For more information on the format, refers to the dataset folder *LF_example/* and the *LF_datasets_example.txt* file
* A yaml configuration file to set up the training parameters. *Config/UnrolledADMMFDL_5x5_2fs.yaml* is an example of a configuration file.


### Command line

The following command line is an example of how to launch the training:
    
    
    python main.py --training_dataset training_datasets.txt --validation_dataset validation_datasets.txt --config Configs/UnrolledADMMFDL_5x5_2fs.yaml --model_name my_model_name --mode train.
    

The model *my_model_name* will be saved in the *Models/* directory.

## Testing

### Preparation

Before launching the testing, you need to prepare the following files:
* A testing dataset file listing the path to the corresponding dataset folders. For more information on the format, refers to the dataset folder *LF_example* and the *LF_datasets_example.txt* file
* A yaml configuration file to set up the testing parameters. *Config/UnrolledADMMFDL_5x5_2fs.yaml* is an example of a configuration file.
* A trained model located in the *Models/* repertory.

### Command line

The following command line is an example of how to launch the testing to reconstruct a light field:

    python main.py --testing_dataset testing_datasets.txt --config Configs/UnrolledADMMFDL_5x5_2fs.yaml --model_name my_model_name --mode test --save_directory save_directory_folder --output_type views
The following command line is an example of how to launch the testing to reconstruction a set of refocus images:

    python main.py --testing_dataset testing_datasets.txt --config Configs/UnrolledADMMFDL_5x5_2fs.yaml --model_name my_model_name --mode test --save_directory save_directory_folder --output_type FS
    
The model *my_model_name* in the *Models/* directory will be used, and the results will be saved in the *save_directory_folder* folder.