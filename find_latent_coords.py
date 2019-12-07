import json
import os
import glob

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image

from models import BetaVAE, Decoder

# Now import the parameters.
with open("params.json") as json_file:  
    params = json.load(json_file)
    dataset = params["dataset"]
    batch_size = params["batch_size"]
    lr = params["lr"]
    test_batch_size = params["test_batch_size"]
    epochs = params["epochs"]
    seed = params["seed"]
    use_cuda = params["use_cuda"]
    display_step = params["display_step"]
    save_step = params["save_step"]
    basepath = params["basepath"]
    conds = params["conds"]
    z_dim = params["z_dim"]
    
# Function to manage the training of the decoder.
def train(model, const, image, device, optimizer, epoch):
    model.train()
    optimizer.zero_grad()
    output = model(const)
    loss = model.loss(output, image)
    loss.backward()
    optimizer.step()
    return loss, output

# Function to fetch the parameters of the VAE model.
def fetch_params(save_file):
    if torch.cuda.is_available():
        restore_state_dict = torch.load(save_file)
    else:
        restore_state_dict = torch.load(save_file, map_location='cpu')
    return restore_state_dict

# Main analysis loop.
if __name__=='__main__':
    
    # Load example holdout and test images for each training condition.
    csv_path = basepath + 'CSVs/'
    holdout_csvs = ['green_squares_heldout.csv', 'lower_right_green_squares_heldout.csv', 'lower_right_square_heldout.csv', 'upper_right_red_heldout.csv']
    holdout_images = [pd.read_csv(csv_path+path).loc[13].filename for path in holdout_csvs]
    #test_csvs = ['green_squares_test.csv', 'lower_right_green_squares_test.csv', 'lower_right_square_test.csv', 'upper_right_red_test.csv']
    #test_images = [pd.read_csv(csv_path+path).loc[12].filename for path in test_csvs]
    
    # Now tune parameters for each condition.
    counter = 0    
    for cond in conds:
        betas = conds[cond]
        heldout_path = holdout_images[counter]
        #test_path = test_images[counter]
        heldout_image = Image.open(basepath + 'Data/' + heldout_path)
        #test_image = Image.open(basepath+'Data/'+test_path)
        counter += 1
        
        # Tune all models and save their final weights (weights will correspond to the latent space coordinates for the heldout image).
        for beta in betas:
            # Set up the file paths that we will use to save the tuned model weights and the final images.
            model_name = "VAE_B" + str(int(100*beta)) + "_Z" + str(z_dim)
            ckpt_dir = basepath + "Checkpoints/" + cond + "/" + model_name + "/"
            output_dir = basepath + "Output/" + cond + "/" + model_name + "/Evaluation/" 
            os.makedirs(output_dir, exist_ok=True)
            
            # Load the model weights.
            checkpoint = sorted(glob.glob(ckpt_dir + '/*.pt'), key=os.path.getmtime)[0]
            device = 'cpu'
            model = Decoder().to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Load the most recent model and set up the constant node (z_dim connections between this node and the bottleneck layer). 
            params = fetch_params(checkpoint)           
            const = torch.tensor(1.0).unsqueeze(0).float()
            tensor = transforms.functional.to_tensor(heldout_image)
            
            # Train the decoder to tune the first z_dim parameters.
            i_epoch = 0
            loss, initial_output = train(model, const, tensor, device, optimizer, i_epoch)
            while loss > 400:
                model.restore_partial(params)
                loss, output = train(model, const, tensor, device, optimizer, i_epoch)
                i_epoch += 1
                if i_epoch%1000==0:
                    print(i_epoch)
            
            # Save the images.
            tuned_img = transforms.functional.to_pil_image(output.squeeze())
            initial_img = transforms.functional.to_pil_image(initial_output.squeeze())
            tuned_img.save(output_dir+'tuned.png')
            initial_img.save(output_dir+'initial.png')
            
            # Save the final weights.
            weights = model.single.weight.detach().numpy()
            np.savetxt(output_dir+'weights.txt', weights, delimiter=',')
            
            # Save the model
            model.save_model(ckpt_dir + 'decoder.pt')
            
       
