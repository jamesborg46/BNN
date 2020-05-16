import torch
import torch.nn as nn
from torch import optim
from torch.nn import init
import torch.nn.functional as F

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt

import datetime
import time
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

        if bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features).cuda())
            self.bias_rho = nn.Parameter(torch.Tensor(out_features).cuda())

        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_rho, a=math.sqrt(5))
        if self.bias_mu is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_mu, -bound, bound)
            init.uniform_(self.bias_rho, -bound, bound)

    def forward(self, input):
        self.weight_sigma = torch.log(1 + torch.exp(self.weight_rho))
        weight = (torch.distributions
                       .Normal(self.weight_mu, self.weight_sigma)
                       .rsample())

        if self.bias_mu is not None:
            self.bias_sigma = torch.log(1 + torch.exp(self.bias_rho))
            bias = (torch.distributions
                         .Normal(self.bias_mu, self.bias_sigma)
                         .rsample())

            return F.linear(input, weight, bias)
        else:
            return F.linear(input, weight, None)

    def kl_loss(self, weight_prior_sigma, bias_prior_sigma):
        weight_kl_loss = 0.5 * torch.sum(
            (self.weight_mu + self.weight_sigma**2) / (weight_prior_sigma**2)
            - 1 - torch.log(self.weight_sigma**2)
            + math.log(weight_prior_sigma**2)
        )

        if self.bias_mu is not None:
            bias_kl_loss = 0.5 * torch.sum(
                (self.bias_mu + self.bias_sigma**2) / (bias_prior_sigma**2)
                - 1 - torch.log(self.bias_sigma**2)
                + math.log(bias_prior_sigma**2)
            )

        return weight_kl_loss + bias_kl_loss


class BayesianNN(nn.Module):

    def __init__(self,
                 weight_prior_sigma,
                 bias_prior_sigma,
                 activation_function=None):

        super(BayesianNN, self).__init__()

        self.weight_prior_sigma = weight_prior_sigma
        self.bias_prior_sigma = bias_prior_sigma

        self.dense_variational_1 = DenseVariational(784, 1200)
        self.dense_variational_2 = DenseVariational(1200, 1200)
        self.dense_variational_3 = DenseVariational(1200, 10)
        if activation_function is None:
            self.activation_function = F.relu
        else:
            self.activation_function = activation_function

    def forward(self, input):
        x = torch.flatten(input, 1)
        x = self.activation_function(self.dense_variational_1(x))
        x = self.activation_function(self.dense_variational_2(x))
        x = self.activation_function(self.dense_variational_3(x))
        return x

    def kl_loss(self,):
        kl_loss = 0
        for child in self.children():
            kl_loss += child.kl_loss(self.weight_prior_sigma,
                                     self.bias_prior_sigma)
        return kl_loss

    def sample(self, input, num_samples):
        samples = []
        logits = []
        for _ in range(num_samples):
            x = self.forward(input)
            logits.append(x)
            x = F.softmax(x, dim=1)
            samples.append(x)
        samples = torch.stack(samples)
        preds = torch.mean(samples, dim=0)
        samples = torch.transpose(samples, 0, 1)
        logits = torch.stack(logits)
        return preds, {"sampled_probs": samples, "sampled_logits": logits}


def train(model, device, train_loader, optimizer, epoch, samples=3):
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
                (1/dataset_size) * complexity_cost
                + likelihood_cost
            )

            average_complexity_cost += (1/samples) * (1/dataset_size) * complexity_cost
            average_likelihood_cost += (1/samples) * likelihood_cost
            average_loss += loss

        average_loss.backward()
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
            pred, info = model.sample(data, 50)
            # complexity_cost = model.kl_loss()
            # likelihood_cost = F.cross_entropy(pred,
            #                                   target,
            #                                   reduction='mean')
            # loss += (1/num_batches) * (
            #     (1/dataset_size) * complexity_cost + likelihood_cost
            # )
            correct += (torch.argmax(pred, 1) == target).sum().item()
            total += len(data)

    test_accuracy = 100 * (correct / total)
    logger.info('Test set: Average loss: {:.4f}\n Accuracy: {:.4f}'
                .format(test_loss, test_accuracy))
    # return loss, test_accuracy
    return test_accuracy


def cuda_timer(func, *args, **kwargs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    a = time.time()
    results = func(*args, **kwargs)
    b = time.time()
    end.record()

    torch.cuda.synchronize()

    cuda_time = start.elapsed_time(end) / 1000

    logger.info("CUDA: {} took {} secs to end"
                .format(func.__name__, cuda_time))

    return results, cuda_time


def main():
    parser = argparse.ArgumentParser(description='PyTorch BNN')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--custom-init', action='store_true', default=False)
    parser.add_argument('--weight-mu-mean-init', type=float, default=0.1)
    parser.add_argument('--weight-mu-scale-init', type=float, default=0.1)
    parser.add_argument('--weight-rho-mean-init', type=float, default=0.1)
    parser.add_argument('--weight-rho-scale-init', type=float, default=0.1)
    parser.add_argument('--bias-mu-mean-init', type=float, default=0.1)
    parser.add_argument('--bias-mu-scale-init', type=float, default=0.1)
    parser.add_argument('--bias-rho-mean-init', type=float, default=0.1)
    parser.add_argument('--bias-rho-scale-init', type=float, default=0.1)
    parser.add_argument('--weight-prior', type=float, default=0.1)
    parser.add_argument('--bias-prior', type=float, default=0.1)
    parser.add_argument('--samples', type=int, default=3)
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
        num_workers=4,
    )

    test_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=args.test_batch_size,
        shuffle=True,
        num_workers=8,
    )

    example_test_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=5,
        shuffle=True,
        num_workers=1,
    )

    example_test = iter(example_test_loader)

    logger.info("Training set size: {}".format(len(train_loader.dataset)))
    logger.info("Test set size: {}".format(len(test_loader.dataset)))

    model = BayesianNN(
        weight_prior_sigma=args.weight_prior,
        bias_prior_sigma=args.bias_prior,
        activation_function=F.elu
    ).to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

    wandb.watch(model, log="all")

    def weights_init(m):
        if isinstance(m, DenseVariational):
            init.normal_(m.weight_mu,
                         args.weight_mu_mean_init,
                         args.weight_mu_scale_init)
            init.normal_(m.weight_rho,
                         args.weight_rho_mean_init,
                         args.weight_rho_scale_init)
            if m.bias_mu is not None:
                init.normal_(m.bias_mu,
                             args.bias_mu_mean_init,
                             args.bias_mu_scale_init)
                init.normal_(m.bias_rho,
                             args.bias_rho_mean_init,
                             args.bias_rho_scale_init)

    if args.custom_init:
        model.apply(weights_init)

    for epoch in range(args.epochs):
        (train_loss,
         train_complexity_cost,
         train_likelihood_cost), train_time = cuda_timer(train,
                                                         model,
                                                         device,
                                                         train_loader,
                                                         optimizer,
                                                         epoch,
                                                         samples=args.samples,
                                                        )

        # test_loss, test_accuracy = test(model, device, test_loader, epoch)
        test_accuracy, test_time = cuda_timer(test,
                                              model,
                                              device,
                                              test_loader,
                                              epoch)

        with torch.no_grad():

            example_data, example_target = example_test.next()
            pred, info = model.sample(example_data.to(device), 30)
            example_samples = info["sampled_probs"]
            # example_preds = torch.argmax(example_logits, 1)
            # example_softmax = F.softmax(example_logits, dim=1)

            plts = []
            for i, example in enumerate(example_data):
                fig, (ax1, ax2) = plt.subplots(2,
                                               figsize=(4, 8),
                                               gridspec_kw={
                                                   'height_ratios': [2, 1]
                                               })
                ax1.imshow(example[0], cmap="gray")
                ax2.violinplot(example_samples.cpu().numpy()[i],
                               positions=range(10),
                               showmeans=True)
                ax2.set_ylim([0, 1])
                plts.append(fig)

            wandb.log(
                {"train_loss": train_loss,
                 "train_complexity_cost": train_complexity_cost,
                 "train_likelihood_cost": train_likelihood_cost,
                 # "test_loss": test_loss,
                 "test_accuracy": test_accuracy,
                 "example_imgs": [wandb.Image(
                     fig,
                     # caption="Logits: {}, Probs: {}".format(
                     #     str(example_logits.cpu()[i]),
                     #     str(example_samples.cpu()[i])
                     # )
                 ) for fig in plts],
                 # "example_imgs":
                 #     [wandb.Image(
                 #         example_data[i],
                 #         caption="Pred: {}\n Probs: {}\n Logits: {}"
                 #         .format(example_preds[i],
                 #                 str(example_softmax[i].cpu()),
                 #                 str(example_logits[i].cpu()))
                 #     ) for i in range(len(example_data))],
                 # "example_logits": example_logits.cpu(),
                 "train_time": train_time,
                 "test_time": test_time,
                 "epoch": epoch+1
                 })
            plt.close('all')


if __name__ == '__main__':
    main()
