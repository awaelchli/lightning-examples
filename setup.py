from setuptools import setup

setup(
    name="lightning-examples",
    version="0.0.1",
    author="Adrian Walchli",
    # packages=["lightning_examles"],
    description="PyTorch examples powered by Lightning.",
    license="MIT",
    install_requires=[
        "torch",
        "lightning",
    ],
)
