from torch.nn import Module, Sequential, LeakyReLU, Linear, Tanh, BatchNorm1d


class Generator(Module):
    def __init__(self, z_dimension, img_size):
        super(Generator, self).__init__()

        self.generator = Sequential(
            Linear(z_dimension, 128),
            BatchNorm1d(128),
            LeakyReLU(.1),
            Linear(128, 256),
            BatchNorm1d(256),
            LeakyReLU(.1),
            Linear(256, img_size),
            Tanh(),
        )

    def forward(self, x):
        return self.generator(x)
