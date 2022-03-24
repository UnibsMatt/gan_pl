from gan.model.gan import Gan
from pytorch_lightning.trainer import Trainer
from gan.datasets.mnist_datamodule import MnistDm
import torch


def run_training():
    """
    Script used to run the training
    """

    # Gan model with generator and discriminator
    gan_model = Gan()

    # Mnist datamodule
    gan_data_module = MnistDm()

    gpus = 1 if torch.cuda.is_available() else None
    trainer = Trainer(
        gpus=gpus,
        progress_bar_refresh_rate=100,
        min_epochs=5,
        max_epochs=100,
        num_sanity_val_steps=1,
    )
    # run trainer
    trainer.fit(gan_model, datamodule=gan_data_module)


if __name__ == '__main__':
    run_training()