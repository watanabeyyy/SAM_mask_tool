"""MASK Predictor using SAM"""

from __future__ import annotations

import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def get_mask(anns: list) -> np.ndarray:
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3)])
        img[m] = color_mask
    return img

class MaskGenerator():
    def __init__(self, device:str) -> None:
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=torch.device(device))
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.3,
            stability_score_thresh=0.3,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
        )

    def pred(self, image: np.ndarray) -> np.ndarray:
        masks = self.mask_generator.generate(image)
        if len(masks) == 0:
            raise ValueError("no mask detected.")
        mask_image = get_mask(masks)
        return mask_image

if __name__=="__main__":
    device = "cuda"
    m = MaskGenerator(device)
    impath = "img/test.png"
    image = cv2.imread(impath)
    resize_mag = 1
    image = cv2.resize(image, (int(image.shape[1]*resize_mag), int(image.shape[0]*resize_mag)))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = m.pred(image)

    cv2.imshow("",image)
    cv2.waitKey(0)
