import numpy as np

def paraGeneration(options):
    np.random.seed(2001)
    cpu_frequency = np.round(np.random.uniform(0.1, 3, size = options['num_of_clients']), decimals=1)
    # eer = [round(np.random.normal(1, 0.2, ), 2) for _ in range(args.nc)]
    # remainEngery = [round(np.random.normal(1, 0.2, ), 2) for _ in range(args.nc)]
    B = np.round(np.random.randint(1, 20, size = options['num_of_clients']), decimals=1)
    transmitpower = [0.5 for _ in range(options['num_of_clients'])]
    g_N0 = [8 for _ in range(options['num_of_clients'])]
    return cpu_frequency, B, transmitpower, g_N0

