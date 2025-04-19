import os
import argparse
from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

# simple cnn model definition
# I looked a lot at https://github.com/giusarno/SimpleCNN/blob/master/examples/cifar10/themodel.py
# before making this class, mostly because I was not aware of the `MaxPool2d` function


class SimpleCNN(nn.Module):
	def __init__(self, num_classes=10):
		super(SimpleCNN, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
			nn.MaxPool2d(2),
		)
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
			nn.Linear(128, num_classes),
		)

	def forward(self, x):
		x = self.features(x)
		x = self.classifier(x)
		return x


def get_data_loaders(batch_size, augmentation):
	# transform pipelines
	if augmentation == 'none':
		transform_train = transforms.Compose([
			transforms.ToTensor(),
		])
	elif augmentation == 'standard':
		transform_train = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomCrop(32, padding=4),
			transforms.ToTensor(),
		])
	elif augmentation == 'aggressive':
		transform_train = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation(15),
			transforms.RandomCrop(32, padding=4),
			transforms.ColorJitter(brightness=0.2, contrast=0.2,
			                       saturation=0.2, hue=0.1),
			transforms.ToTensor(),
		])
	else:
		raise ValueError(f"unknown augmentation: {augmentation}")

	transform_test = transforms.Compose([
		transforms.ToTensor(),
	])

	train_dataset = torchvision.datasets.CIFAR10(
		root='./data', train=True, download=True, transform=transform_train)
	test_dataset = torchvision.datasets.CIFAR10(
		root='./data', train=False, download=True, transform=transform_test)

	train_loader = DataLoader(
		train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
	test_loader = DataLoader(
		test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

	return train_loader, test_loader


# train for 1 epoch


def train_one_epoch(model, optimizer, criterion, dataloader, device):
	model.train()
	running_loss = 0.0
	correct = 0
	total = 0
	for inputs, targets in dataloader:
		inputs, targets = inputs.to(device), targets.to(device)
		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()

		running_loss += loss.item() * inputs.size(0)
		_, predicted = outputs.max(1)
		correct += predicted.eq(targets).sum().item()
		total += targets.size(0)
	epoch_loss = running_loss / total
	epoch_acc = correct / total
	return epoch_loss, epoch_acc


# eval on clean data


def evaluate(model, criterion, dataloader, device):
	model.eval()
	running_loss = 0.0
	correct = 0
	total = 0
	with torch.no_grad():
		for inputs, targets in dataloader:
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = model(inputs)
			loss = criterion(outputs, targets)

			running_loss += loss.item() * inputs.size(0)
			_, predicted = outputs.max(1)
			correct += predicted.eq(targets).sum().item()
			total += targets.size(0)
	loss = running_loss / total
	acc = correct / total
	return loss, acc


# eval robustness under gaussian noise


def evaluate_robustness(model, dataloader, device, noise_std):
	model.eval()
	correct = 0
	total = 0
	with torch.no_grad():
		for inputs, targets in dataloader:
			noisy_inputs = inputs + noise_std * torch.randn_like(inputs)
			noisy_inputs = torch.clamp(noisy_inputs, 0.0, 1.0)
			noisy_inputs, targets = noisy_inputs.to(device), targets.to(device)
			outputs = model(noisy_inputs)
			_, predicted = outputs.max(1)
			correct += predicted.eq(targets).sum().item()
			total += targets.size(0)
	acc = correct / total
	return acc


def analyze_results(results_path='results.json'):
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.formula.api import ols
    import statsmodels.api as sm

    with open(results_path) as f:
        results = json.load(f)
    df = pd.DataFrame(results)
    df.to_csv('analysis_results.csv', index=False)

    # full ANOVA w/interaction
    model = ols('test_acc ~ C(optimizer) * C(augmentation)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print('anova on test accuracy:')
    print(anova_table)

    # composite label
    df['condition'] = df['optimizer'] + '_' + df['augmentation']
    df.plot.bar(x='condition', y='test_acc', rot=45)
    plt.ylabel('test accuracy')
    plt.tight_layout()
    plt.savefig('test_acc_comparison.png')
    print('saved plot to test_acc_comparison.png')


# main (PUBLIC STATIC VOID AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA)
def run_experiments(args):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	results = []

	optimizers = {
		'sgd': lambda params: optim.SGD(params, lr=args.lr, momentum=0.9),
		'adam': lambda params: optim.Adam(params, lr=args.lr)
	}
	augmentations = ['none', 'standard', 'aggressive']

	seeds = [42, 123, 999]
	for seed in seeds:
		torch.manual_seed(seed)
		np.random.seed(seed)
		for opt_name in optimizers:
			for aug in augmentations:
				train_loader, test_loader = get_data_loaders(args.batch_size, aug)
				model = SimpleCNN(num_classes=10).to(device)
				optimizer = optimizers[opt_name](model.parameters())
				criterion = nn.CrossEntropyLoss()
				history = {
					'epoch': [], 'train_loss': [], 'train_acc': [],
					'test_loss': [], 'test_acc': []
				}

				for epoch in range(args.epochs):
					train_loss, train_acc = train_one_epoch(
						model, optimizer, criterion, train_loader, device)
					test_loss, test_acc = evaluate(
						model, criterion, test_loader, device)

					history['epoch'].append(epoch + 1)
					history['train_loss'].append(train_loss)
					history['train_acc'].append(train_acc)
					history['test_loss'].append(test_loss)
					history['test_acc'].append(test_acc)

					print(f"[{opt_name}][{aug}][epoch {epoch + 1}] "
                                            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                                            f"test_acc={test_acc:.4f}")

				noise_levels = [0.1, 0.2, 0.3]
				robustness = {noise: evaluate_robustness(
					model, test_loader, device, noise) for noise in noise_levels}

				pd.DataFrame(history).to_csv(
                                    f"history_{opt_name}_{aug}_{seed}.csv", index=False)
				results.append({
					'seed': seed,
					'optimizer': opt_name,
					'augmentation': aug,
					'test_acc': test_acc,
					'robustness': robustness
				})

	with open('results.json', 'w') as f:
		json.dump(results, f, indent=2)
	print('saved results to results.json')


# credit: I gave chatgpt a list of args and it made the arg parser for me
if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--lr', type=float, default=0.01)
	parser.add_argument('--epochs', type=int, default=10)
	parser.add_argument('--analyze', action='store_true',
	                    help='run analysis on results')

	args = parser.parse_args()

	if args.analyze:
		analyze_results()
	else:
		run_experiments(args)
