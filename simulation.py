import torch
import torch.nn as nn
import torch.optim as optim

from GMgenerator import GenerateSet
from nets import Perceptron
from results import Results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Code associated to the MSc project entitled
'Occurence of Neural Collapse on Gaussian mixture classification with deep neural networks'
by Alexandre de Skowronski

Check out the report to understance the numerical setup
if any questions may arise e-mail me at
alexdesko@gmail.com
"""

class Simulation():
    """
    Class to define a simluation to run
    - N nb of points
    - sigma size of all the clouds
    - input_dim the dim in which the points lie one
    - num_classes the number of classes
    - hidden_dim the hidden dimensions of the classifiers
    - depth the depth of the network (if depth = L then L+1 layers!!)
    - criterionstr the criterion to be used in string format
    - batch_size pretty straighforward right ?
    - lr learning rate for training
    - wd weight decay value
    - epochs total of epochs
    - two_bulbs bbool to say whether one wants one or two bulbs per class
    - rho centroid distacnes from the origin
    - simplex to replace or no the last layer of the classifier by an ETF


    """
    def __init__(self, N, sigma, input_dim, num_classes, hidden_dim, depth, criterionstr, batch_size, lr, wd, epochs,two_bulbs = True, rho = 1, simplex = False):

        self.name = 'N{}_s{}_dim{}_nc{}_h{}_d{}_lr{:.0e}_wd{:.0e}_'.format(N, sigma, input_dim, num_classes, hidden_dim, depth, lr, wd)
        self.name += '_batch{}_epochs{}'.format(batch_size, epochs)

        self.epochs = epochs
        self.num_classes = num_classes

        if two_bulbs == False:
            self.name += '_onebulb'
        if rho != 1:
            self.name += '_rho{}'.format(rho)
        if simplex == True:
            self.name += '_2'

        print('Initiate simulation ', self.name)

        torch.manual_seed(42)

        _, train_loader, bayes_train = GenerateSet(N, num_classes, input_dim, sigma, batch_size, two_bulbs, rho)
        _, test_loader, bayes_test = GenerateSet(N, num_classes, input_dim, sigma, batch_size, two_bulbs, rho)

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.model = Perceptron(input_dim, hidden_dim, num_classes, depth, simplex).to(device)
        print(self.model)
        if criterionstr == 'MSE':
            self.criterion = nn.MSELoss()
            self.criterion_analysis = nn.MSELoss()
        elif criterionstr == 'CrossEntropyLoss':
            self.criterion = nn.CrossEntropyLoss()
            self.criterion_analysis = nn.CrossEntropyLoss()
        else:
            pass

        self.optimizer= optim.SGD(self.model.parameters(), lr = lr, momentum = .9, weight_decay=wd)
        self.results = Results(self.name)

        self.results.metrics["bayes_train"].append(1 - bayes_train.cpu().item())
        self.results.metrics["bayes_test"].append(1 - bayes_test.cpu().item())



    def run(self):
        for epoch in (range(1, self.epochs+1)):
            self.model.train_loop(self.optimizer, self.criterion, self.train_loader, self.criterion2)
            if (epoch) % 500 == 0 or epoch < 100:
                self.results.SaveResults(self.model, self.criterion_analysis, self.train_loader, self.test_loader, epoch, self.num_classes, True)

        self.results.ToCSV(self.name + '/')

