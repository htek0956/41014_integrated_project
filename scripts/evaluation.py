class Logger:


    def __init__(self):
        raise NotImplementedError("Initialise the experiment logs.")

    def record(self, timestamp, estimate, reference, command, truth):

        raise NotImplementedError("Store an evaluation sample.")

    def results(self):

        raise NotImplementedError("Calculate experiment metrics.")
