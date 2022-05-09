from simulation import Simulation

if __name__ == "__main__":
    """
    Example of run when running in the cluster
    From experience, and with the code I've wrote, running the simulations in parallel is not drastically beneficial
    """
    p = 100
    Npoints = 1000
    input_dim = 1000
    nc = [2]  
    sigma = .4
    epochs = 2
    wd = [0]
    batch_size = 10
    depth = [1]
    lr = 5e-3

    bool_two_bulbs = False

    for _, num_classes in enumerate(nc):    
        for i, wd_ in enumerate(wd):
            for _, d in enumerate(depth):
                for criterionstr in ['MSE', 'CrossEntropyLoss']:
                    Sim = Simulation(Npoints, sigma, input_dim, num_classes, p, d, criterionstr, batch_size, lr, wd_, epochs, two_bulbs= bool_two_bulbs, rho = 1, simplex=False)
                    Sim.run()
                    del Sim