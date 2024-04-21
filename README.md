# General information
Codalab username: Jakub_Kasprzyk

TU/e email: j.m.kasprzyk@student.tue.nl

The neccessary checkpoint files required for running the visualizations and model evaluations in `model_test.ipynb` cna be found in the following OneDrive shared folder: https://tuenl-my.sharepoint.com/:f:/r/personal/j_m_kasprzyk_student_tue_nl/Documents/5LSM0-JKasprzyk-checkpointfiles/model_checkpoints?csf=1&web=1&e=CV7ywM


# Final Assignment

This repository contains the completed project for the 5LSM0 final assignment.
This assignment is a part of the 5LSM0 course. It involves working with the Cityscapes dataset and training a neural network. The projectfocus on obtaining a Robust segmentaiton model through means of exploring different variations of U-Net models, implementation of different objective functions and Data Augmentation.

## Getting Started

### Dependencies

We already created a DockerContainer with all dependencies to run on Snellius, in the run_main.sh file we refer to this container. You don't have to changes anything for this.

### Installing

To get started with this project, you need to clone the repository to Snellius or your local machine. You can do this by running the following command in your terminal:

```bash
https://github.com/JMKasprzyk/FinalAssignment.git
```

After cloning the repository, navigate to the project directory:

```bash
cd FinalAssignment
```

All the implementation utilize standard PyTorch libraries not additional instalations needed. In the `train.py` all the avaliable criterions and imported model architectures are commented. For ease of use you can comment out any model or criterion of choice and train on it.

### File Descriptions

Here's a brief overview of the files you'll find in this repository:

- **`run_container.sh`:** Contains the script for running the container. In this file you have the option to enter your wandb keys if you have them and additional arguments if you have implemented them in the train.py file.

  
- **`run_main`:** Includes the code for building the Docker container. In this file, you only need to change the settings SBATCH (the time your job will run on the server) and ones you need to put your username at the specified location.
  

- **`model.py`:** Defines the neural network architecture.

  
- **`train.py`:** Contains the code for training the neural network.

- **`metrics.py`** Contains implementes scorring metrics.

- **`losses.py`** Contains implemented loss functinos.

- **`model_executables.py`** Crucial script containing all the traing loop function definitinos.

- **`model_vis.py`** Contains functions for visualizing the sematnic segmenation performed by a given to the funcitno model.

- **`MS_UNet.py`** Multi-scale U-Net implementaion.

- **`ResUNet.py`** Residual U-Net implementaiton.

- **`Att_UNet.py`** Attetnion U-Net implementaiton.

- **`Res_Att_UNet.py`** Residual Attention U-Net implementaion.

- **`R2_UNet.py`** Recurrent Residual U-Net implementation. (requires further work)

- **`R2Att_UNet.py`** Recurrent Residual Attention U-Net implementation. (requires further work)

- **`model_test.ipynb`** Jupyter notebook intended for visualization of models' semangic segmentaiton performance.

### Authors

- T.J.M. Jaspers
- C.H.B. Claessens
- C.H.J. Kusters
