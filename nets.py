import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ETF(input_dim, num_classes):
    """
    Function to generate a simplex ETF
    Mainly use as a utility, for example to replace the last layer of a network with an ETF

    returns a tensor
    """
    ETF = torch.sqrt(torch.tensor(num_classes/(num_classes-1)))*(torch.eye(num_classes)-(1/num_classes)*torch.ones((num_classes, num_classes)))
    ETF /= torch.norm(ETF, 2, dim=0)
    if input_dim != num_classes:
        # multiply on the right because of the implementation of nn.Linear
        ETF = torch.mm(ETF, torch.eye(num_classes, input_dim)) 
    return ETF

class Perceptron(nn.Module):
    """
    Class that represents a deep classifier, specified by:
    - input_dim the dim of the inputs
    - hidden_dim the hidden dimensions
    - output_dim the number of classes to fit
    - depth the numbber of hidden layer (if depth=L the L+1 total layers)
    - simplex a boolean to specify if we want the last layer to be a simplex ETF
    """
    def __init__(self, input_dim, hidden_dim, output_dim, depth=1, simplex = False):
        super(Perceptron, self).__init__()
        self.output_dim = output_dim
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(1, depth):
            self.fcs.append(nn.Linear(hidden_dim, hidden_dim))
        self.classifier = nn.Linear(hidden_dim, output_dim)
        # If the last layer is a simplex, lower the penultime dimension to put a square simplex
        # all of this assuming that hidden_dim >= output_dim holds (which makes sense so it always should)
        if simplex:
            self.fcs[-1] = nn.Linear(hidden_dim, output_dim)
            self.classifier = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1).float()
        for _, fc in enumerate(self.fcs):
            x = F.relu(fc(x))
        with torch.no_grad():
            features = x.clone()
        return self.classifier(x).squeeze(), features.squeeze()

    def train_loop(self, optimizer, criterion, train_loader):
        """
        Method to perform an epoch in training, taking as input:
        - the optimizer defined outside
        - the criterion used
        - the train_loader loading the data
        """
        self.train()
        for _, (x_, y_) in enumerate(train_loader):
            x_, y_ = x_.to(device), y_.to(device)
            optimizer.zero_grad()
            out, _ = self.forward(x_)
            if str(criterion) == 'CrossEntropyLoss()':
                loss = criterion(out.squeeze(), y_.long().squeeze())
            elif str(criterion) == 'MSELoss()':
                loss = criterion(out.squeeze(), F.one_hot(y_.long(), num_classes=self.output_dim).float().squeeze())   
            else:
                raise NotImplementedError()
            loss.backward()
            optimizer.step()


def model_test():
    model = Perceptron(10,10,10,True,1)
    print(model)
    model2 = Perceptron(5, 10, 3, True, 3)
    print(model2)

if __name__ == "__main__":
    model_test()