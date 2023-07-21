# 3D-Dental-Mesh-Segmentation-Using-Semantics-Based-Feature-Learning-with-Graph-Transformer
Codes for MICCAI 2023 paper: 3D Dental Mesh Segmentation Using Semantics-Based Feature Learning with Graph-Transformer

## Time to Open our code

7.21 when the camera-ready files are submitted.

## Project dependencies

python~=3.7

pytorch==1.11.0+cu113

vtk==9.2.2

vedo==2022.4.1

The more detailed dependencies can be checked in the `requirements.txt`.

## Project Configuration

First, you need to install all the libraries listed in the `requirements.txt`.

```shell
pip install -r requirements.txt
```

To train your network, you need to specify all the `xxxxx` in the `train.py` to specify your data loader and log directory. In detail, the input of our network is an $N\times24$ matrix for each mesh. The initial $N \times 12$ C-domain matrix composes of the 3D coordinates of the 3 vertices and the centroid of each cell, and the initial $N \times 12$ N-domain matrix composes of the normal vectors of the 3 vertices and the centroid of each cell. Then you can run

```shell
python train.py
```

and your network can be trained and the tested data will be listed in your specified directory.

## Dataset

We are sorry, but due to our business agreement with our partners, we are unable to provide the data. Please prepare the data yourself for training the network.





