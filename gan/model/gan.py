import pytorch_lightning as pl
from gan.model.discriminator import Discriminator
from gan.model.generator import Generator
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import BCELoss
from torchvision.utils import save_image
import torch


class Gan(pl.LightningModule):
    def __init__(self,
                 image_dimension: int = 784,
                 z_dim: int = 64,
                 lr: float = 1e-4,
                 batch_size: int = 64,
                 use_cuda: bool = True,
                 ):
        super(Gan, self).__init__()
        self.use_cuda = use_cuda & torch.cuda.is_available()
        self.image_dimension = image_dimension
        self.z_dim = z_dim
        self.lr = lr
        self.discriminator = Discriminator(image_dimension)
        self.generator = Generator(img_size=image_dimension, z_dimension=z_dim)

        self.fixed_noise = torch.rand((batch_size, z_dim))
        self.loss = BCELoss()

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_images, l = batch
        real_images = real_images.view(-1, 784)
        batch_size = real_images.shape[0]

        valid = torch.ones(real_images.size(0), 1).to(self.device)
        fake = torch.zeros(real_images.size(0), 1).to(self.device)

        if optimizer_idx == 0:
            fake_input = torch.randn(batch_size, self.z_dim).to(self.device)
            self.generated_images = self(fake_input)
            g_loss = self.loss(self.discriminator(self.generated_images), valid)
            return {"loss": g_loss}
        else:
            real_loss = self.loss(self.discriminator(real_images), valid)
            fake_loss = self.loss(self.discriminator(self.generated_images.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2.

            return {"loss": d_loss}

    def configure_optimizers(self):
        optimizer_generator = Adam(self.generator.parameters(), self.lr, betas=(0.4, 0.99))
        optimizer_discriminator = Adam(self.discriminator.parameters(), self.lr, betas=(0.4, 0.99))
        lr_scheduler_d = StepLR(optimizer_discriminator, step_size=50, gamma=.5)
        lr_scheduler_g = StepLR(optimizer_generator, step_size=50, gamma=.5)

        return [optimizer_generator, optimizer_discriminator], [lr_scheduler_d, lr_scheduler_g]

    def on_epoch_end(self) -> None:
        save_image(self.generated_images[10:30].reshape(-1, 1, 28, 28), fp=f"img/{self.current_epoch}.png", normalize=True, nrow=5)




