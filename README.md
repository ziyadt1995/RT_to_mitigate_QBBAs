# RT_to_mitigate_QBBAs
This repository provides code for our paper "Random Transformations to Mitigate Query-Based Black-Box Attacks"

We have modified the codebase of BlackBox Bench (https://github.com/SCLBD/BlackboxBench) and the Robustness Library (https://github.com/MadryLab/robustness).

Once you have installed the robustness library please replace the entire files with the files we have provided to successfully run the code without errors. 

The ResNet-50 models can be obtained from the Robustness Library (https://github.com/MadryLab/robustness). The ImageNet test images can be retrieved from TREMBA (https://github.com/TransEmbedBA/TREMBA).

We provide the CIFAR-10 and Tiny ImageNet shuffled images in our repository. 

We recommend running the code on SPYDER IDE using Anaconda (https://www.anaconda.com/). 

To launch attacks, run the files by selecting their code "as selection" in Spyder IDE. Use the following files:

1. attack_imagenet.py
2. attack_cifar10.py
3. attack_tinyimagenet.py

Run the command "pip install -r requirementsnew.txt" to create the required Python environment with the necessary packages. 

NOTE: The copyrights of the code used in this repository belong to their respective creators.
