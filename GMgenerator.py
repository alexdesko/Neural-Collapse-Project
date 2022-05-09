import torch
from torch.utils.data import DataLoader, Dataset

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class to embed dataset
class CustomSet(Dataset):
    def __init__(self, input, target):
        self.input =  input.to(device)
        self.target = target.to(device)
    def __len__(self):
        return(len(self.input))
    def __getitem__(self, idx):
        return (self.input[idx], self.target[idx])

# Generate a single class
def GenerateClass(centroids, sigma, Ntot, label):
    """
    Function to generate a class given:
    - the centroids
    - sigma (std) of all the bulbs
    - Ntot the total number of points in the class
    - label

    returns the features and the label in tensors
    """
    if len(centroids.size()) ==1: # case of only one centroid
        centroids.unsqueeze_(0)
    Ncentroids = centroids.size(0)
    N = Ntot // Ncentroids
    x = torch.normal(centroids.repeat(N,1), sigma)
    y = label*torch.ones(size=(Ntot,1))
    

    return x.to(device), y.to(device) 

# Generate the entire set, returning a dataloader
def GenerateSet(N, nc, dim, sigma = 1, batch_size=10, two_bulbs = True, rho = 1):
    """
    Function to generate a dataset given:
    - the number of points per class N
    - the number of classes nc
    - the std of all the bulbs sigma
    - the batch size when constructing the data loader
    - the boolean tw_bulbs to say whether one or two clouds inn wanted per class
    - rho the distance of the centroids to the origin

    return the set embeded in an instance of the class CustomSet
    and the dataloader associated
    """
    centroids_1 = rho * torch.eye(dim)[:nc]
    if two_bulbs:
        centroids_2 =  -centroids_1
    else:
        centroids_2 = centroids_1
    x = torch.FloatTensor(size=(N*nc, dim)).to(device)
    y = torch.LongTensor(size=(N*nc, 1)).to(device)

    for lbl in range(nc):
        x[N*lbl:N*(lbl+1), :], y[N*lbl:N*(lbl+1), :] = GenerateClass(torch.cat([centroids_1[lbl,:].unsqueeze(-1), centroids_2[lbl,:].unsqueeze(-1)], dim=1).T, sigma, N, lbl)

    y.squeeze_()
    set = CustomSet(x, y)
    
    if two_bulbs:
        bayes = torch.cosh(x[:,:nc] / sigma**2)
        bayes_error = torch.mean((torch.argmax(bayes, dim=1) != y).float())
    else:
        bayes = torch.exp( x[:,:nc] / sigma**2)
        bayes_error = torch.mean((torch.argmax(bayes, dim=1) != y).float())

    
    return set, DataLoader(set, batch_size=batch_size, shuffle = True), bayes_error