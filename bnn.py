import torch
import torch.nn as nn
import torch.nn.functional as F

# from torchvision import datasets
# from torchvision import transforms

# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import numpy as np

from utils import batch_linear

# import datetime
# import argparse
import math


class DenseVariational(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(DenseVariational, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(
            torch.Tensor(out_features, in_features))

        self.weight_rho = nn.Parameter(
            torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features).cuda())
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).cuda())

    def reset_parameters(self,
                         weight_mu_mean,
                         weight_mu_scale,
                         weight_rho_mean,
                         weight_rho_scale,
                         bias_mu_mean,
                         bias_mu_scale,
                         bias_rho_mean,
                         bias_rho_scale,
                         ):

        nn.init.normal_(self.weight_mu,
                        weight_mu_mean,
                        weight_mu_scale)

        nn.init.normal_(self.weight_rho,
                        weight_rho_mean,
                        weight_rho_scale)

        nn.init.normal_(self.bias_mu,
                        bias_mu_mean,
                        bias_mu_scale)

        nn.init.normal_(self.bias_rho,
                        bias_rho_mean,
                        bias_rho_scale)

    def forward(self, input):
        samples, batch_size, units = input.shape

        self.weight_sigma = torch.log(1 + torch.exp(self.weight_rho))

        weight = (torch.distributions
                       .Normal(self.weight_mu, self.weight_sigma)
                       .rsample((samples,)))

        self.bias_sigma = torch.log(1 + torch.exp(self.bias_rho))
        bias = (torch.distributions
                     .Normal(self.bias_mu, self.bias_sigma)
                     .rsample((samples,)))

        return batch_linear(input, weight, bias)

    def kl_loss(self, weight_prior_sigma, bias_prior_sigma):
        weight_kl_loss = 0.5 * torch.sum(
            (self.weight_mu**2 + self.weight_sigma**2) / (weight_prior_sigma**2)
            - 1 - torch.log(self.weight_sigma**2)
            + math.log(weight_prior_sigma**2)
        )

        bias_kl_loss = 0.5 * torch.sum(
            (self.bias_mu**2 + self.bias_sigma**2) / (bias_prior_sigma**2)
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

    def reset_parameters(self,
                         **kwargs):
        self.dense_variational_1.reset_parameters(**kwargs)
        self.dense_variational_2.reset_parameters(**kwargs)
        self.dense_variational_3.reset_parameters(**kwargs)

    def forward(self, input, samples=1):
        batch, features = input.shape
        x = input.repeat(samples, 1, 1)
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
