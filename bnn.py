import torch
import torch.nn as nn
from torch import optim
from torch.nn import init
import torch.nn.functional as F

from torchvision import datasets
from torchvision import transforms

import datetime
import argparse
import math

import logging

import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DenseVariational(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(DenseVariational, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(
            torch.Tensor(out_features, in_features))

        self.weight_rho = nn.Parameter(
            torch.Tensor(out_features, in_features))

        self.register_buffer(
            'weight_sigma',
            torch.log(1 + torch.exp(self.weight_rho))
        )

        self.weight_dist = torch.distributions.Normal(
            self.weight_mu,
            self.weight_sigma,
        )

        self.register_buffer(
            'weight_signal_to_noise',
            torch.abs(self.weight_mu) / self.weight_sigma
        )

        self.register_buffer(
            'weight_entropy',
            0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.weight_sigma)
        )

        if bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_rho = nn.Parameter(torch.Tensor(out_features))

            self.register_buffer(
                'bias_sigma',
                torch.log(1 + torch.exp(self.bias_rho))
            )

            self.bias_dist = torch.distributions.Normal(
                self.bias_mu,
                self.bias_sigma
            )

            self.register_buffer(
                'bias_signal_to_noise',
                torch.abs(self.bias_mu) / self.bias_sigma
            )

            self.register_buffer(
                'bias_entropy',
                0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.bias_sigma)
            )
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
            self.register_buffer('bias_sigma', None)
            self.register_buffer('bias_signal_to_noise', None)
            self.register_buffer('bias_entropy', None)
            self.bias_dist = None

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_std, a=math.sqrt(5))
        if self.bias_dist is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_mu, -bound, bound)
            init.uniform_(self.bias_std, -bound, bound)

    def forward(self, input):
        weight = self.weight_dist.rsample()
        if self.bist_dist is not None:
            bias = self.bias_dist.rsample()
            return F.linear(input, weight, bias)
        else:
            return F.linear(input, weight, None)

    def kl_loss(self, weight_prior_sigma, bias_prior_sigma):
        weight_kl_loss = 0.5 * torch.sum(
            (self.weight_mu + self.weight_sigma**2) / (weight_prior_sigma**2)
            - 1 - torch.log(self.weight_sigma**2)
            + math.log(weight_prior_sigma**2)
        )

        bias_kl_loss = 0.5 * torch.sum(
            (self.bias_mu + self.bias_sigma**2) / (bias_prior_sigma**2)
            - 1 - torch.log(self.bias_sigma**2)
            + math.log(bias_prior_sigma**2)
        )

        return weight_kl_loss + bias_kl_loss


class BayesianNN(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 weight_prior_sigma,
                 bias_prior_sigma,
                 activation_function=None):

        super(BayesianNN, self).__init__()
        self.dense_variational_1 = DenseVariational(in_features, 1200)
        self.dense_variational_2 = DenseVariational(1200, 1200)
        self.dense_variational_3 = DenseVariational(1200, 10)
        if activation_function is None:
            self.activation_function = F.relu
        else:
            self.activation_function = activation_function

    def forward(self, input):
        x = self.activation_function(self.dense_variational_1(input))
        x = self.activation_function(self.dense_variational_2(x))
        x = self.activation_function(self.dense_variational_3(x))
        return x

    def kl_loss(self,):
        kl_loss = 0
        for child in self.children():
            kl_loss += child.kl_loss(self.weight_prior_sigma,
                                     self.bias_prior_sigma)
        return kl_loss


def train(model, device, train_loader, optimizer, epoch, samples=1):
    model.train()
    num_training_steps = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        batch_size = len(data)
        dataset_size = len(train_loader.dataset)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        average_complexity_cost = 0
        average_likelihood_cost = 0
        average_loss = 0
        for _ in range(samples):
            pred = model(data)
            complexity_cost = model.kl_loss()
            likelihood_cost = F.cross_entropy(pred,
                                              target,
                                              reduction='mean')
            loss = (1/samples) * (
                (1/dataset_size) * complexity_cost + likelihood_cost
            )
            loss.backward()

            average_complexity_cost += (1/samples) * complexity_cost
            average_likelihood_cost += (1/samples) * likelihood_cost
            average_loss += loss

        optimizer.step()

        num_training_steps += batch_size

        if batch_idx % 200 == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * batch_size, dataset_size,
                        100. * batch_idx / len(train_loader), loss.item()))

    return average_loss, average_complexity_cost, average_likelihood_cost


def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    dataset_size = len(test_loader.dataset)
    num_batches = len(test_loader)
    loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data)
            complexity_cost = model.kl_loss()
            likelihood_cost = F.cross_entropy(pred,
                                              target,
                                              reduction='mean')
            loss += (1/num_batches) * (
                (1/dataset_size) * complexity_cost + likelihood_cost
            )
            correct += (torch.argmax(pred, 1) == target).sum().item()
            total += len(data)

    test_accuracy = 100 * (correct / total)
    logger.info('Test set: Average loss: {:.4f}\n Accuracy: {:.4f}'
                .format(test_loss, test_accuracy))
    return loss, test_accuracy


def main():
    parser = argparse.ArgumentParser(description='PyTorch BNN')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--custom-init', action='store_true', default=False)
    parser.add_argument('--weight-mu-init', type=float, default=0.01)
    parser.add_argument('--weight-rho-init', type=float, default=0.01)
    parser.add_argument('--bias-mu-init', type=float, default=0.01)
    parser.add_argument('--bias-rho-init', type=float, default=0.01)
    parser.add_argument('--weight-prior', type=float, default=0.001)
    parser.add_argument('--bias-prior', type=float, default=0.3)
    parser.add_argument('--pre-normalization',
                        action='store_true', default=False)

    args = parser.parse_args()

    logger.info(args)

    exp_name = (datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
                + '_' + args.name)

    wandb.init(
        name=exp_name,
        project="bnn-project",
        config=args,
    )

    device = torch.device('cuda:0')

    if args.pre_normalization:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    train_data = datasets.MNIST('./mnist/', transform=transform)
    validation_data = datasets.MNIST('./mnist/', train=False,
                                     transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8
    )

    test_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=args.test_batch_size,
        shuffle=True,
        num_workers=8,
    )

    logger.info("Training set size: {}".format(len(train_loader.dataset)))
    logger.info("Test set size: {}".format(len(test_loader.dataset)))

    model = BayesianNN().to(device)

    optimizer = optim.Adagrad(lr=args.lr)

    wandb.watch(model, log="all")

    def weights_init(m):
        if isinstance(m, DenseVariational):
            init.normal_(m.weight_mu, 0.0, args.weight_mu_init)
            init.normal_(m.weight_rho, 0.0, args.weight_rho_init)
            if m.bias_dist is not None:
                init.normal_(m.bias_mu, 0.0, args.bias_mu_init)
                init.normal_(m.bias_rho, 0.0, args.bias_rho_init)

    if args.custom_init:
        model.apply(weights_init)

    for epoch in range(args.epochs):
        (train_loss,
         train_complexity_cost,
         train_likelihood_cost) = train(model,
                                        device,
                                        train_loader,
                                        optimizer,
                                        epoch)

        test_loss, test_accuracy = test(model, device, test_loader, epoch)

        with torch.no_grad():
            wandb.log(
                {"train_loss": train_loss,
                 "train_complexity_cost": train_complexity_cost,
                 "train_likelihood_cost": train_likelihood_cost,
                 "test_loss": test_loss,
                 "test_accuracy": test_accuracy,
                 "epoch": epoch+1
                 })


if __name__ == '__main__':
    main()
