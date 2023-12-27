import os, sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tinysam import sam_model_registry, SamPredictor
from tinysam.quantization_layer import InferQuantMatMul, InferQuantMatMulPost, InferQuantLinear, InferQuantLinearPost, InferQuantConv2d, InferQuantConvTranspose2d
from demo import show_mask, show_points, show_box

model_type = "vit_t"
net_ = sam_model_registry[model_type]()

cpt_path = os.path.join(sam_path, "weights/tinysam_w8a8.pth")
quant_sam = torch.load(cpt_path)                                   
predictor = SamPredictor(quant_sam)

image = cv2.imread('fig/picture1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)
input_point = np.array([[365, 225]])
input_label = np.array([1])
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

plt.figure(figsize=(10,10))
plt.imshow(image)
show_mask(masks[scores.argmax(),:,:], plt.gca())
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.savefig("test_quant.png")