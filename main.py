import torch
import torch.nn.functional as F

from torchvision import datasets
from torchvision import transforms

from bnn import BayesianNN
from utils import cuda_timer
from train import train, test, active_sampling
from results_logger import ResultsLogger

import numpy as np
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
    parser.add_argument('--mu-excluded', action='store_true', default=False)
    parser.add_argument('--explicit-gradient', action='store_true',
                        default=False)
    parser.add_argument('--empirical-complexity', action='store_true',
                        default=False)
    parser.add_argument('--active-sampling', action='store_true', default=False)
    parser.add_argument('--active-samples', type=int, default=5)
    parser.add_argument('--iters-between-active-samples', type=int, default=10)
    parser.add_argument('--initial-samples', type=int, default=60000)
    parser.add_argument('--final-samples', type=int, default=60000)
    parser.add_argument('--initial-iterations', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=5000)
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
    parser.add_argument('--prior-mix', type=float, default=1.0)
    parser.add_argument('--weight-prior', type=float, default=0.1)
    parser.add_argument('--bias-prior', type=float, default=0.1)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--pre-normalization',
                        action='store_true', default=False)

    args = parser.parse_args()

    logger.info(args)

    exp_name = (args.name + '_'
                + datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))

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

    subset_indices = np.random.choice(len(train_data),
                                      args.initial_samples,
                                      replace=False)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=8,
        sampler=torch.utils.data.SubsetRandomSampler(subset_indices)
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
        activation_function=F.elu,
        prior_mix=args.prior_mix,
        empirical_complexity_loss=args.empirical_complexity,
        explicit_gradient=args.explicit_gradient,
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
                                               epoch,
                                               samples=args.samples)

        test_results, test_time = cuda_timer(test,
                                             model,
                                             device,
                                             test_loader,
                                             results_logger)

        results_logger.log_epoch(epoch, train_time, test_time)

        # print("{} {} {} {}".format(args.active_sampling,
                                # len(train_loader.sampler.indices),
                                # args.initial_iterations,
                                # args.iters_between_active_samples))

                # len(train_loader.sampler.indices) < args.final_samples and \
        if args.active_sampling and \
                epoch > args.initial_iterations and \
                epoch % args.iters_between_active_samples == 0:

            train_loader = active_sampling(model,
                                           device,
                                           train_data,
                                           train_loader,
                                           results_logger,
                                           args.active_samples,
                                           )


if __name__ == '__main__':
    main()
