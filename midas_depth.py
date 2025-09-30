import torch
import cv2
import numpy as np
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet

def load_midas_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
    model_path = torch.hub.load("intel-isl/MiDaS", "DPT_BEiT_L_512", trust_repo=True)
    model = DPTDepthModel(
        path=model_path,
        backbone="beitl16_512",
        non_negative=True
    )
    model.eval()
    model.to(device)

    transform = Resize(
        512, 512,
        resize_target=None,
        keep_aspect_ratio=True,
        ensure_multiple_of=32,
        resize_method="minimal",
        image_interpolation_method=cv2.INTER_CUBIC,
    )
    transform = Compose([
        transform,
        NormalizeImage(mean=[0.5]*3, std=[0.5]*3),
        PrepareForNet()
    ])

    return model, transform, device

def estimate_depth(image, model, transform, device):
    image_input = transform({"image": image})["image"]
    input_tensor = torch.from_numpy(image_input).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model.forward(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()
    return depth
