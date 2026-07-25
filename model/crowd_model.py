import torch
from PIL import Image
import torchvision.transforms as transforms

import matplotlib

# Prevent Flask/Tkinter matplotlib error
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


from model.csrnet import CSRNet



# Load CSRNet model

model = CSRNet()


checkpoint = torch.load(
    "model/weights.pth",
    map_location="cpu",
    weights_only=False
)


model.load_state_dict(
    checkpoint["state_dict"]
)


model.eval()





# Image transformation

transform = transforms.Compose([
    transforms.ToTensor()
])






def create_heatmap(density_map):

    density = density_map.squeeze().cpu().numpy()


    plt.figure(figsize=(6,4))


    plt.imshow(
        density,
        cmap="jet"
    )


    plt.axis("off")



    plt.savefig(
        "static/uploads/heatmap.png",
        bbox_inches="tight",
        pad_inches=0
    )


    plt.close()








def predict_crowd(image_path):


    print("Loading image:", image_path)



    # Load image

    image = Image.open(image_path).convert("RGB")



    image = transform(image)



    image = image.unsqueeze(0)





    # Prediction

    with torch.no_grad():

        density_map = model(image)





    # Count people

    count = torch.sum(density_map)


    people = int(count.item())





    # Create heatmap

    create_heatmap(density_map)





    # Crowd classification

    if people < 50:


        level = "LOW CROWD"


        message = "The image contains a small number of people."



    elif people < 150:


        level = "MEDIUM CROWD"


        message = "The image contains a moderate crowd."



    else:


        level = "HIGH CROWD"


        message = "The image contains a dense crowd."







    # Temporary confidence score

    confidence = 92





    print("Predicted people:", people)

    print("Crowd level:", level)





    return {


        "count": people,


        "level": level,


        "message": message,


        "heatmap": "static/uploads/heatmap.png",


        "confidence": confidence

    }