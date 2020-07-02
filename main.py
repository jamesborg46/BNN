import torch
import torch.nn.functional as F

from torchvision import datasets
from torchvision import transforms

from bnn import BayesianNN
from utils import cuda_timer
from train import train, test
from results_logger import ResultsLogger

import datetime
import argparse

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='PyTorch BNN')
    parser.add_argument('--name', type=str, default="UNAMED")
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
        shuffle=False,
        num_workers=8,
    )

    logger.info("Training set size: {}".format(len(train_loader.dataset)))
    logger.info("Test set size: {}".format(len(test_loader.dataset)))

    model = BayesianNN(
        weight_prior_sigma=args.weight_prior,
        bias_prior_sigma=args.bias_prior,
        activation_function=F.elu
    ).to(device)

    # parameter initialization
    model.reset_parameters(
         weight_mu_mean=args.weight_mu_mean_init,
         weight_mu_scale=args.weight_mu_scale_init,
         weight_rho_mean=args.weight_rho_mean_init,
         weight_rho_scale=args.weight_rho_scale_init,
         bias_mu_mean=args.bias_mu_mean_init,
         bias_mu_scale=args.bias_mu_scale_init,
         bias_rho_mean=args.bias_rho_mean_init,
         bias_rho_scale=args.bias_rho_scale_init,
    )

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    results_logger = ResultsLogger(exp_name,
                                   model,
                                   args)

    for epoch in range(args.epochs):
        train_results, train_time = cuda_timer(train,
                                               model,
                                               device,
                                               train_loader,
                                               optimizer,
                                               results_logger,
                                               samples=args.samples)

        test_results, test_time = cuda_timer(test,
                                             model,
                                             device,
                                             test_loader,
                                             results_logger)

        results_logger.log_epoch(epoch, train_time, test_time)

if __name__ == '__main__':
    main()
