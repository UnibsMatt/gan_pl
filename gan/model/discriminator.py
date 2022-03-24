from torch.nn import Module, Sequential, LeakyReLU, Linear, Sigmoid


class Discriminator(Module):
    def __init__(self, in_feature):
        super(Discriminator, self).__init__()

        self.discriminator = Sequential(
            Linear(in_feature, 128),
            LeakyReLU(.1),
            Linear(128, 256),
            LeakyReLU(.1),
            Linear(256, 1),
            Sigmoid(),
        )

    def forward(self, x):
        return self.discriminator(x)





