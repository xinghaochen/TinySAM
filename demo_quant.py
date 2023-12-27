import os, sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tinysam import sam_model_registry, SamPredictor
from demo import show_mask, show_points, show_box

sys.path.append("./tinysam")

cpt_path = "weights/tinysam_w8a8.pth"
quant_sam = torch.load(cpt_path) 
device = "cuda" if torch.cuda.is_available() else "cpu"
quant_sam.to(device=device)                                 
predictor = SamPredictor(quant_sam)

image = cv2.imread('fig/picture1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)
input_point = np.array([[365, 225]])
input_label = np.array([1])
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
)

plt.figure(figsize=(10,10))
plt.imshow(image)
show_mask(masks[scores.argmax(),:,:], plt.gca())
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.savefig("test_quant.png")