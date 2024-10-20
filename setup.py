from setuptools import setup, find_packages

setup(
    name="dlo_perceiver",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "transformers",
        "wandb",
        "albumentations",
        "matplotlib",
        "wandb",
        "tqdm",
        "shutils",
        "einops",
        "h5py",
    ],
)
