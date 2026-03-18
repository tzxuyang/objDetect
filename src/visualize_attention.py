import os
import sys
import argparse
from turtle import width
# from PIL.IcnsImagePlugin import height
import cv2
import random
import colorsys
from matplotlib import axes
from regex import W
import requests
from io import BytesIO
import timm

# import skimage.io
# from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image, ImageOps
import torch.nn.functional as F
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

import utils
from dino_train import DinoClassifier, set_seed, train_classifier

NUM_CLASS = 3
PATH = "./checkpoints/dino_classifier.pth"
IMAGE_PATH = "./images/pick_n_place_1.jpg"
# IMAGE_PATH = "./images/dog1.jpg"
# IMAGE_PATH = "./images/cat4.jpg"
WIDTH, HEIGHT = 640, 480
# WIDTH, HEIGHT = 480, 640
PATCH_SIZE = 16

# def visualize_attention(attentions, img_path, output_path):
def show_img(img):
    img = np.asarray(img)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load model
    model = DinoClassifier(num_classes=NUM_CLASS)
    model.load_state_dict(torch.load(PATH))

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)

    print("Done loading model ")
    print(model)

    # open image
    img = Image.open(IMAGE_PATH).convert("RGB")
    data_config = timm.data.resolve_model_data_config(model.backbone)
    height, width = data_config['input_size'][1], data_config['input_size'][2]
    print("data_config: ", data_config)
    # img_size = (WIDTH, HEIGHT)
    img_tensor = model.process_image(data_config, IMAGE_PATH).to(device)
    print("image shape is:")
    print(img_tensor.shape)

    # get attention map
    for block in model.backbone.blocks:
        block.attn.fused_attn = False
    nodes, _ = get_graph_node_names(model.backbone)
    interesting_nodes = [x for x in nodes if 'attn_drop' in x]
    print("Interesting nodes: ", interesting_nodes)

    feature_extractor = create_feature_extractor(model.backbone, return_nodes=interesting_nodes)
    out = feature_extractor(img_tensor)

    for k, v in out.items():
        print(k, v.shape)

    num_layers = len(interesting_nodes)
    num_heads = model.backbone.blocks[0].attn.num_heads
    print(f"Number of layers: {num_layers}, Number of heads: {num_heads}")

    fig, axs = plt.subplots(1, 2, figsize=(16, 16))
    for i, (k, v) in enumerate(out.items()):
        if i == num_layers - 1:
            attn_scores = v.squeeze()
            attn_scores_mean = attn_scores.mean(dim=0)
            cls_token_attn_scores = attn_scores_mean[0,5:]
            axs[1].imshow(cls_token_attn_scores.reshape(height // PATCH_SIZE, width // PATCH_SIZE).detach().cpu().numpy(), cmap='viridis')
    axs[0].imshow(img)
    for ax in axs.flat:
        ax.set(xticks=[], yticks=[])
    plt.show()