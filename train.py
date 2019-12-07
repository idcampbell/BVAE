import time
import json
import os
import multiprocessing

import torch
import torch.optim as optim
from torchvision.utils import save_image
from torchvision import transforms
from torch.utils.data import DataLoader

import utils
from dataset import ShapesDataset
from models import BetaVAE


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
    
    
# Function to manage the training of the network.
def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    epoch_total_loss = 0
    epoch_recon_loss = 0
    epoch_KLD = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output, mu, logvar = model(data)
        total_loss, recon_loss, KLD = model.loss(output, data, mu, logvar)
        total_loss.backward()
        optimizer.step()
        epoch_total_loss += total_loss.item()
        epoch_recon_loss += recon_loss.item()
        epoch_KLD += KLD.item()
        if batch_idx % log_interval == 0:
            print("{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f}".format(time.ctime(time.time()), epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), recon_loss.item()))
    epoch_total_loss /= len(train_loader)
    epoch_recon_loss /= len(train_loader)
    epoch_KLD /= len(train_loader)
    print("Train set average total loss: ", epoch_total_loss)
    print("Train set average reconstruction loss: ", epoch_recon_loss)
    print("Train set average KLD: ", epoch_KLD)
    print("\n")
    return (epoch_total_loss, epoch_recon_loss, epoch_KLD)


# Function to manage the testing of the network.
def test(model, device, test_loader, return_images=0, log_interval=None):
    model.eval()
    epoch_total_loss = 0
    epoch_recon_loss = 0
    epoch_KLD = 0
    original_images = []
    rect_images = []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            output, mu, logvar = model(data)
            total_loss, recon_loss, KLD = model.loss(output, data, mu, logvar)
            epoch_total_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_KLD += KLD.item()
            if return_images > 0 and len(original_images) < return_images:
                original_images.append(data[0].cpu())
                rect_images.append(output[0].cpu())
            if log_interval is not None and batch_idx % log_interval == 0:
                print("{} Test: [{}/{} ({:.0f}%)]\tLoss: {:.2f}".format(time.ctime(time.time()), batch_idx * len(data), len(test_loader.dataset), 100. * batch_idx / len(test_loader), recon_loss.item()))
    epoch_total_loss /= len(test_loader)
    epoch_recon_loss /= len(test_loader)
    epoch_KLD /= len(test_loader)
    print("Test set average total loss: ", epoch_total_loss)
    print("Test set average reconstruction loss: ", epoch_recon_loss)
    print("Test set average KLD: ", epoch_KLD)
    print("\n")
    return (epoch_total_loss, epoch_recon_loss, epoch_KLD), original_images, rect_images


# Set some seeds.
use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {"num_workers": multiprocessing.cpu_count(), "pin_memory": True} if use_cuda else {} #multiprocessing.cpu_count()
print('Using device', device)
print('num cpus:', multiprocessing.cpu_count())

# Set up GPU training and some training parameters.
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Execute model training and testing.
if __name__ == "__main__":
    
    for cond in conds:
        betas = conds[cond]

        # Set up the Datasets and the DataLoaders.
        transform = transforms.Compose([transforms.ToTensor()])
        train_csv = basepath + "CSVs/" + cond + "_train.csv"
        test_csv = basepath + "CSVs/" + cond + "_test.csv"
        train_dataset = ShapesDataset(train_csv, transform=transform)
        test_dataset = ShapesDataset(test_csv, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, **kwargs)

        for beta in betas:
            model_name = "VAE_B" + str(int(100*beta)) + "_Z" + str(z_dim)
            ckpt_dir = basepath + "Checkpoints/" + cond + "/" + model_name + "/"
            log_dir = basepath + "Logs/" + cond + "/" + model_name + "/"
            log = log_dir + "log.pkl" 
            output_dir = basepath + "Output/" + cond + "/" + model_name + "/Training/"
            os.makedirs(ckpt_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            print("Training CSV: " + train_csv)
            print("Testing CSV: " + test_csv)
            print("Checkpoint Directory: " + ckpt_dir)
            print("Log Directory: " + log_dir)
            print("Output Directory: " + output_dir)

            # Instantiate the model and optimizer.
            model = BetaVAE(beta).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Load the most recent model and log.
            start_epoch = model.load_last_model(ckpt_dir) + 1
            train_losses, test_losses = utils.read_log(log, ([], []))

            # Set test_losses to a dummy tuple.
            if start_epoch == 1:
                 test_losses = [ [1, (1000, 1000, 1000)] ]

            # Begin training model.
            for epoch in range(start_epoch, epochs + 1):
                if test_losses[-1][1][1] > 300:
                    epoch_train_losses = train(model, device, train_loader, optimizer, epoch, display_step)
                    epoch_test_losses, original_images, rect_images = test(model, device, test_loader, return_images=5)
                    save_image(original_images + rect_images, output_dir + str(epoch) + '.png', padding=0, nrow=len(original_images))
                    train_losses.append((epoch, epoch_train_losses))
                    test_losses.append((epoch, epoch_test_losses))
                    utils.write_log(log, (train_losses, test_losses))
                    model.save_model(ckpt_dir + '%03d.pt' % epoch)
