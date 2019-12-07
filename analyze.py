import os
import json

import numpy as np
import pandas as pd
import PIL
import torch
from torchvision.utils import save_image
import torchvision.transforms.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

import utils
from model import BetaVAE
from dataset import ShapesDataset


with open("params.json") as json_file:  
    params = json.load(json_file)
    basepath = params["basepath"]

ckpt_dir = basepath + 'Checkpoints/'
output_dir = basepath + 'Output/'
csv_dir = basepath + 'CSVs/'
data_dir = basepath + 'Data/'


# Upper left hand corner circles.
#img1 = data_dir + "circle_x10_y10_size14_215_red.png"
#img2 = data_dir + "circle_x10_y10_size14_215_green.png"
#img3 = data_dir + "circle_x10_y10_size14_215_blue.png"
# Lower left hand corner squares.
img1 = data_dir + "circle_x41_y31_size14_137_green.png" #"square_x53_y53_size12_129_red.png"
img5 = data_dir + "square_x53_y53_size14_129_green.png"
img6 = data_dir + "square_x53_y53_size14_129_blue.png"
# Lower left hand corner circles.
img7 = data_dir + "circle_x53_y53_size14_225_red.png"
img8 = data_dir + "circle_x53_y53_size14_225_green.png"
img9 = data_dir + "circle_x53_y53_size14_225_blue.png"


# Load the base image and generate some interpolations.
img = PIL.Image.open(img1)
tensor = transforms.functional.to_tensor(img)

# This function finds the z vector encoding of a given image.
def get_z(im, model, device):
    model.eval()
    im = torch.unsqueeze(im, dim=0).to(device)
    with torch.no_grad():
        mu, var = model.encode(im)
        z = model.reparameterize(mu, var)
    return z

# Interpolates along a specified dimension of a given z vector.
def interpolate_z(z, model, i):
    model.eval()
    factors = np.linspace(-2, 2, num=25, dtype=float)
    result = []
    with torch.no_grad():
        for f in factors:
             z = z.clone()
             z[0][i] = torch.as_tensor(f).to("cpu")
             im = torch.squeeze(model.decode(z).cpu())
             result.append(im)
    return result


# Generate interpolations for all dimensions of a given vector z.
def interpolate_all_zs(z, model):
    results = []
    for i in range(10):
        results.append(interpolate_z(z, model, i))
    return results


# Loads a model, saves a grid like image of the interpolations (should only work for model where z_dim==10),
# and also saves gifs of the interpolations.
def checkpoint_model(model, tensor, output_dir):
    z = get_z(tensor, model, "cpu")
    interpolations = interpolate_all_zs(z, model)
    flat_list = [item for sublist in interpolations for item in sublist]
    save_image(flat_list, output_dir+'grid.png', nrow=10)
    counter = 1
    for interpolation in interpolations:
        gif_path = output_dir + '/'
        interpolation = [F.to_pil_image(pic) for pic in interpolation]
        os.makedirs(gif_path, exist_ok=True)
        interpolation[0].save(output_dir+str(counter)+".gif", save_all=True, append_images=interpolation[1:], duration=100, loop=0)
        counter += 1


# Evaluates the reconstruction loss.
def evaluate_loss(model, csv):
    transform = transforms.Compose([transforms.ToTensor()])
    df = pd.read_csv(csv, index_col=0).reset_index()
    dataset = ShapesDataset(csv, transform=transform)
    loader = DataLoader(dataset, batch_size=100, shuffle=True)
    loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data = data.to("cpu")
            output, mu, logvar = model(data)
            total, recon, kld = model.loss(output, data, mu, logvar)
            loss += recon.item()
    return loss / len(df)


# This function finds the beta parameter that was used to train the model.
def find_beta(model_string):
    return int(model_string.split('_')[1].split('B')[1]) / 100
 
   
# Now locate the models, save the interpolation gifs/grids, and evaluate the losses.
for train_cond in os.listdir(ckpt_dir):
    cond_dir = os.path.join(ckpt_dir, train_cond)
    print(train_cond)
    for model_name in os.listdir(cond_dir):
        model_path = os.path.join(cond_dir, model_name)
        try:
            for model_ckpt in os.listdir(model_path):
                if model_ckpt.endswith(".pt"):
                    test_csv = csv_dir+train_cond+'_test.csv'
                    heldout_csv = csv_dir+train_cond+'_heldout.csv'
                    output_path = output_dir+train_cond+'\\'+model_name+'\\'
                    beta = find_beta(model_name)
                    model = BetaVAE(beta).to("cpu")
                    model.load_last_model(model_path)
                    checkpoint_model(model, tensor, output_path)
                    test_loss = evaluate_loss(model, test_csv)
                    heldout_loss = evaluate_loss(model, heldout_csv)
                    print("Test loss for " + model_name + " trained on " + train_cond + " :" + str(test_loss)) 
                    print("Heldout loss for " + model_name + " trained on " + train_cond + " :" + str(heldout_loss)) 
        except FileNotFoundError:
            print('FileNotFound')
