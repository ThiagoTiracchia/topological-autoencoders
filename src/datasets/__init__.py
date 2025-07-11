"""Datasets."""
from .manifolds import Spheres, SwissRoll
from .mnist import MNIST
from .fashion_mnist import FashionMNIST
from .cifar10 import CIFAR
__all__ = ['Spheres', 'MNIST', 'FashionMNIST', 'CIFAR','SwissRoll']
