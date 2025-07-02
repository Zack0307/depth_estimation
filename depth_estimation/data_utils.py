import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from torchvision import models, transforms
from PIL import Image as PILImage
from torchvision.models.segmentation import (deeplabv3_resnet50,  DeepLabV3_ResNet50_Weights)


def DeepLabv3(weights):
    model = deeplabv3_resnet50(weights = weights)
    model.eval()
   
        
        

    

