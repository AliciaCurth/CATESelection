class DataGenerator(object):
    def __init__(self):
        # set attributes
        self.name = "generator-vanilla"

    def __call__(self, seed=42):
        raise NotImplementedError("need implemented DGP")
