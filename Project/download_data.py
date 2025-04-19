from torchvision.datasets import CIFAR10, CIFAR100
import tensorflow_datasets as tfds

ds10 = CIFAR10(root='data/', train=True, download=True)
ds100 = CIFAR100(root='data/', train=True, download=True)

ds_c10c = tfds.load('cifar10_corrupted')
