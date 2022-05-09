import os
import torch
from torch import cuda
import torch.nn.functional as F
import pandas as pd
from scipy.sparse.linalg import svds
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Results():
    """
    (Thats a big one)
    The implementation of this class is heavily based on the code provided alongside the article
    'Neural Collapse Under MSE Loss: Proximity to and Dynamics on the Central Path' by
    Han, Payan & Donoho

    Basically all of the metrics of intereste are stored in a large dictionnary of arrays

    Takes only the name that the user wants to asign to the simulation as input
    """
    def __init__(self, name = ''):
        self.name = name

        self.metrics = {}

        self.metrics["bayes_train"] = []
        self.metrics["bayes_test"] = []

        self.metrics["epochs"]= [] #
        # Losses
        self.metrics["train_loss"] = [] #
        self.metrics["test_loss"] = [] #
        # Accuracies
        self.metrics["train_accuracy"] = [] #
        self.metrics["test_accuracy"] = [] #
        # Material to compute NC metrics
        self.metrics["features_mean"] = [] #
        self.metrics["weights_mean"] = [] #

        self.metrics["features_std"] = [] 
        self.metrics["weights_std"] = [] 

        self.metrics["NC1"] = []
        self.metrics["NC2"] = []
        self.metrics["NC3"] = []
        self.metrics["NC4"] = []

        self.metrics["class1_means_dist"] = []
        self.metrics["class2_means_dist"] = []
        self.metrics["class3_means_dist"] = []

    def ToCSV(self, dirpath = 'test/'):
        """
        Writing down the results in a CSV, with the dirpath as input
        """
        if not (os.path.exists(dirpath)):
            print('Creating directory')
            os.mkdir(dirpath)
        for key, metric in self.metrics.items():
            print(key)
            df = pd.DataFrame(metric)
            df.to_csv(dirpath + self.name + '_' + key + '.csv', header = False, index = False)
    
    def FromCSV(self, dirpath = ''):
        """
        Taking data from a folder, with the dirpath as input
        """
        for key, _ in self.metrics.items():
            # The following if statement can be removed, it was just for me to import relevant data faster
            if key != 'class3_means_dist' and key != 'class1_means_dist' and key != 'class2_means_dist':
                df = pd.read_csv(dirpath + self.name + '_' + key + '.csv', header = None)
                self.metrics[key] = df.values
    
    def SaveResults(self, model, criterion_analysis, train_loader, test_loader, epoch, num_classes, bool_feature_distance = False, debug = False):
        """
        Method to store all the relevant NC metrics after an epoch
        I recomment to spend some time understanding this
        Again, heavily inspired by Han, Papyan & Donoho
        """

        self.metrics["epochs"].append(epoch)

        N = [0 for _ in range(num_classes)]
        mean = [0 for _ in range(num_classes)]

        loss = 0
        accuracy = 0
        NCC_match_net = 0
        Sw = 0

        features_tot = []
        for computation in ['Mean', 'Cov', 'ClassMeansDist']:

            if computation == 'Cov' and debug:
                continue

            for batch_idx, (input, target) in enumerate(train_loader):

                input, target = input.to(device), target.to(device)

                out, features = model(input)
                features_tot.append(features)
                net_pred = torch.argmax(out, dim=1)
                if computation == 'Mean':
                    accuracy += torch.sum(target == net_pred).item()
                    if str(criterion_analysis) == 'CrossEntropyLoss()':
                        loss += criterion_analysis(out, target).item()
                    elif str(criterion_analysis) == 'MSELoss()':
                        loss += criterion_analysis(out, F.one_hot(target, num_classes=num_classes).float()).item()

                
                for i in range(num_classes):
                    idx = (target == i).nonzero(as_tuple=True)[0]
                    if len(idx) == 0:
                        continue
                    features_c = features[idx,:]
                    if computation == 'Mean':
                        mean[i] += torch.sum(features_c, dim = 0)
                        N[i] += features_c.size(0)
                    elif computation == 'Cov':
                        # Covariance computation
                        currentFeaturesMean = mean[i]
                        temp = features_c - currentFeaturesMean.unsqueeze(0) # B D
                        b = temp.size(0)
                        for k in range(b):
                            cov = torch.matmul(temp[k,:].unsqueeze(-1), 
                                                temp[k,:].unsqueeze(0))
                            Sw += cov

                       # 2) agreement between prediction and nearest class center
                        NCC_scores = torch.stack([torch.norm(features_c[j,:] - M.T,dim=1) \
                                                for j in range(features_c.shape[0])])
                        NCC_pred = torch.argmin(NCC_scores, dim=1)
                        NCC_match_net += sum(NCC_pred==net_pred[idx]).item()

            if computation == 'Mean':
                for c in range(num_classes):
                    mean[c] /= N[c]
                    M = torch.stack(mean).T
            elif computation == 'Cov':
                Sw /= sum(N)                

        self.metrics["train_loss"].append(loss / sum(N))
        self.metrics["train_accuracy"].append(accuracy / sum(N))
        self.metrics["NC4"].append(1 - NCC_match_net / sum(N))


        if not debug:
            features_tot = torch.cat(tuple(features_tot), dim = 0)

            if bool_feature_distance:
                temp2 = features_tot - mean[0].unsqueeze(0)
                self.metrics['class{}_means_dist'.format(1)].append(torch.norm(temp2, 2, dim=1).cpu().numpy())
                

            weights = model.classifier.weight.detach().clone() # C D !!
            W_norms = torch.norm(weights, 2, dim=1)

            self.metrics["weights_mean"].append(torch.mean(W_norms).cpu().item())
            self.metrics["weights_std"].append(torch.std(W_norms).cpu().item())

            globalMean = torch.mean(M, dim=1)
            MCentered = M - globalMean.unsqueeze(-1)
            Sb = torch.matmul(MCentered, MCentered.T) / num_classes # D D
            Sw = Sw.cpu().numpy()
            Sb = Sb.cpu().numpy()
            eigvec, eigval, _ = svds(Sb, k=num_classes-1)
            inv_Sb = eigvec @ np.diag(eigval**(-1)) @ eigvec.T 
            self.metrics["NC1"].append(np.trace(Sw @ inv_Sb) / num_classes)
                
            M_norms = torch.norm(MCentered, 2, dim = 0) # Keep C norms
            self.metrics["features_mean"].append(torch.mean(M_norms).cpu().item())
            self.metrics["features_std"].append(torch.std(M_norms).cpu().item())
                # NC2 - MAXIMAL EQUIANGULARITY
            def coherence(V): 
                G = V.T @ V
                G += torch.ones((num_classes,num_classes),device=device) / (num_classes-1)
                G -= torch.diag(torch.diag(G))
                return torch.norm(G,1) / (num_classes*(num_classes-1))
            
            NC2 = coherence(weights.T/torch.norm(weights.T, dim=0))
            self.metrics["NC2"].append(NC2.cpu().item())

            # NC3 - SELF - DUALITY
            normalized_M = MCentered / torch.norm(MCentered, 'fro')
            normalized_W = weights.T / torch.norm(weights, 'fro')
            NC3 = torch.norm(normalized_W - normalized_M)**2
            self.metrics["NC3"].append(NC3.cpu().item())


        test_loss = 0
        test_accuracy = 0
        N_test = 0
        # Test metrics
        for _, (input_test, target_test) in enumerate(test_loader):

            input_test, target_test = input_test.to(device), target_test.to(device)
            N_test += input_test.size(0)
            out_test, _ = model(input_test)
            net_pred_test = torch.argmax(out_test, dim=1)
            test_accuracy += torch.sum(target_test == net_pred_test).item()
            if str(criterion_analysis) == 'CrossEntropyLoss()':
                test_loss += criterion_analysis(out_test, target_test).item()
            elif str(criterion_analysis) == 'MSELoss()':
                test_loss += criterion_analysis(out_test, F.one_hot(target_test, num_classes=num_classes).float()).item()
                
        self.metrics["test_loss"].append(test_loss / N_test)
        self.metrics["test_accuracy"].append(test_accuracy / N_test)


def test():
    res = Results('HelloWorld')
    res.metrics["train_loss"] = [1, 2, 3]
    res.ToCSV()

if __name__ == "__main__":
    print('Testing results.py')
    test()