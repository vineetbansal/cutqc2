class CutSolution:
    def __init__(self, *, subcircuits, complete_path_map, num_cuts, counter):
        self.subcircuits = subcircuits
        self.complete_path_map = complete_path_map
        self.num_cuts = num_cuts
        self.counter = counter