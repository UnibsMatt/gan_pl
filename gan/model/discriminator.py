from torch.nn import Module, Sequential, LeakyReLU, Linear, Sigmoid


class Discriminator(Module):
    def __init__(self, in_feature: int):
        """
        Base discriminator Module
        Args:
            in_feature: number of feature in input -> image rgb expected 3
        """
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





