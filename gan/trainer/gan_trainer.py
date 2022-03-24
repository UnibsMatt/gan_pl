from gan.model.gan import Gan
from pytorch_lightning.trainer import Trainer
from gan.datasets.mnist_datamodule import MnistDm


def run_training():

    gan_model = Gan()

    gan_data_module = MnistDm()

    trainer = Trainer(
        gpus=1,
        progress_bar_refresh_rate=100,
        min_epochs=5,
        max_epochs=100,
        num_sanity_val_steps=1,
    )

    trainer.fit(gan_model, datamodule=gan_data_module)


if __name__ == '__main__':
    run_training()