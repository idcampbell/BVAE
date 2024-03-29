import subprocess
import os
import glob
import re
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


def tensor_2_img(tensor):
    return transforms.functional.to_pil_image(tensor)


def grid_2_gif(image_str, output_gif, delay=100):
    """Make GIF from images.
    code from:
        https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python/34555939#34555939
    """
    str1 = 'convert -delay '+str(delay)+' -loop 0 ' + image_str  + ' ' + output_gif
    subprocess.call(str1, shell=True)
     
        
def restore(net, save_file):
    net_state_dict = net.state_dict()
    
    if torch.cuda.is_available():
        restore_state_dict = torch.load(save_file)
    else:
        restore_state_dict = torch.load(save_file, map_location='cpu')
        
    restored_var_names = set()
    
    #print('Restoring:')
    for var_name in restore_state_dict.keys():
        if var_name in net_state_dict:
            var_size = net_state_dict[var_name].size()
            restore_size = restore_state_dict[var_name].size()
            if var_size != restore_size:
                print('') #Shape mismatch for var', var_name, 'expected', var_size, 'got', restore_size)
            else:
                if isinstance(net_state_dict[var_name], torch.nn.Parameter):
                    net_state_dict[var_name] = restore_state_dict[var_name].data
                try:
                    net_state_dict[var_name].copy_(restore_state_dict[var_name])
                    #print(str(var_name) + ' -> \t' + str(var_size) + ' = ' + str(int(np.prod(var_size) * 4 / 10**6)) + 'MB')
                    restored_var_names.add(var_name)
                except Exception as ex:
                    print('While copying the parameter named {}, whose dimensions in the model are'
                          ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                              var_name, var_size, restore_size))
                    raise ex

    ignored_var_names = sorted(list(set(restore_state_dict.keys()) - restored_var_names))
    unset_var_names = sorted(list(set(net_state_dict.keys()) - restored_var_names))
    print('')
    if len(ignored_var_names) == 0:
        print('') #Restored all variables')
    else:
        print('') #Did not restore:\n\t' + '\n\t'.join(ignored_var_names))
    if len(unset_var_names) == 0:
        print('') #print('No new variables')
    else:
        print('') #Initialized but did not modify:\n\t' + '\n\t'.join(unset_var_names))

    #print('Restored %s' % save_file)
    
    
def restore_latest(net, folder):
    """Restores the most recent weights in a folder
    Args:
        net(torch.nn.module): The net to restore
        folder(str): The folder path
    Returns:
        int: Attempts to parse the epoch from the state and returns it if possible. Otherwise returns 0.
    """
    checkpoints = sorted(glob.glob(folder + '/*.pt'), key=os.path.getmtime)
    start_it = 0
    if len(checkpoints) > 0:
        restore(net, checkpoints[-1])
        try:
            start_it = int(re.findall(r'\d+', checkpoints[-1])[-1])
        except:
            pass
    return start_it


def save(net, file_name, num_to_keep=1):
    """Saves the net to file, creating folder paths if necessary.
    Args:
        net(torch.nn.module): The network to save
        file_name(str): the path to save the file.
        num_to_keep(int): Specifies how many previous saved states to keep once this one has been saved.
            Defaults to 1. Specifying < 0 will not remove any previous saves.
    """
    folder = os.path.dirname(file_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(net.state_dict(), file_name)
    extension = os.path.splitext(file_name)[1]
    checkpoints = sorted(glob.glob(folder + '/*' + extension), key=os.path.getmtime)
    print('Saved %s\n' % file_name)
    if num_to_keep > 0:
        for ff in checkpoints[:-num_to_keep]:
            os.remove(ff)


def write_log(filename, data):
    """Pickles and writes data to a file
    Args:
        filename(str): File name
        data(pickle-able object): Data to save
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    pickle.dump(data, open(filename, 'wb'))


def read_log(filename, default_value=None):
    """Reads pickled data or returns the default value if none found
    Args:
        filename(str): File name
        default_value(anything): Value to return if no file is found
    Returns:
        un-pickled file
    """

    if os.path.exists(filename):
        return pickle.load(open(filename, 'rb'))
    return default_value


def show_images(images, titles=None, columns=5, max_rows=5, path=None, tensor=False):
    """
    Args:
        images(list[np.array]): Images to show
        titles(list[string]): Titles for each of the images
        columns(int): How many columns to use in the tiling
        max_rows(int): If there are more than columns * max_rows images,
        only the first n of them will be shown.
    """

    images = images[:min(len(images), max_rows * columns)]

    if tensor:
        images = [np.transpose(im, (1, 2, 0)) for im in images]

    plt.figure(figsize=(20, 10))
    for ii, image in enumerate(images):
        plt.subplot(len(images) / columns + 1, columns, ii + 1)
        plt.axis('off')
        if titles is not None and ii < len(titles):
            plt.title(str(titles[ii]))
        plt.imshow(image)

    if path is not None and os.path.exists(os.path.dirname(path)):
        plt.savefig(path)
    else:
        plt.show()
