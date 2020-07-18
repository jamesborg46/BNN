import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import batch_linear

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

        # There variables hold the latest samples of the weights and biases and
        # will be populated after the first forward run of the layer
        self.weight = None
        self.bias = None

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

        self.weight = (torch.distributions
                       .Normal(self.weight_mu, self.weight_sigma)
                       .rsample((samples,)))

        self.bias_sigma = torch.log(1 + torch.exp(self.bias_rho))
        self.bias = (torch.distributions
                     .Normal(self.bias_mu, self.bias_sigma)
                     .rsample((samples,)))

        return batch_linear(input, self.weight, self.bias)

    def empirical_complexity_loss(self, weight_prior_sigma, bias_prior_sigma):
        weight_log_prob = (torch.distributions
                                .Normal(self.weight_mu, self.weight_sigma)
                                .log_prob(self.weight.detach()))

        bias_log_prob = (torch.distributions
                              .Normal(self.bias_mu, self.bias_sigma)
                              .log_prob(self.bias.detach()))

        self.weight_prior = torch.distributions.Normal(0, weight_prior_sigma)
        self.bias_prior = torch.distributions.Normal(0, bias_prior_sigma)

        weight_prior_log_prob = self.weight_prior.log_prob(self.weight)
        bias_prior_log_prob = self.bias_prior.log_prob(self.bias)

        empirical_complexity_loss = (
            torch.sum(weight_log_prob - weight_prior_log_prob) +
            torch.sum(bias_log_prob - bias_prior_log_prob)
        )

        return empirical_complexity_loss

    def kl_loss(self, weight_prior_sigma, bias_prior_sigma, mu_excluded=False):
    # def kl_loss(self, weight_prior_dist, bias_prior_dist, mu_excluded=False):
        if mu_excluded:
            weight_mu = 0
            bias_mu = 0
        else:
            weight_mu = self.weight_mu
            bias_mu = self.bias_mu

        ### Manual KL Calculations - leaving here for reference and in case 
        ### They are required 

        # weight_kl_loss = 0.5 * torch.sum(
        #     (weight_mu**2 + self.weight_sigma**2) / (weight_prior_sigma**2)
        #     - 1 - torch.log(self.weight_sigma**2)
        #     + math.log(weight_prior_sigma**2)
        # 

        # bias_kl_loss = 0.5 * torch.sum(
        #     (bias_mu**2 + self.bias_sigma**2) / (bias_prior_sigma**2)
        #     - 1 - torch.log(self.bias_sigma**2)
        #     + math.log(bias_prior_sigma**2)
        # )

        # return weight_kl_loss + bias_kl_loss

        weight_prior_dist = torch.distributions.Normal(0, weight_prior_sigma)
        bias_prior_dist = torch.distributions.Normal(0, bias_prior_sigma)

        self.weight_dist = torch.distributions.Normal(self.weight_mu,
                                                      self.weight_sigma)
        self.bias_dist = torch.distributions.Normal(self.bias_mu,
                                                    self.bias_sigma)

        return (torch.sum(torch.distributions.kl_divergence(self.weight_dist, weight_prior_dist)) +
                torch.sum(torch.distributions.kl_divergence(self.bias_dist, bias_prior_dist)))


class BayesianNN(nn.Module):

    def __init__(self,
                 weight_prior_sigma,
                 bias_prior_sigma,
                 activation_function=None,
                 prior_mix=1,
                 empirical_complexity_loss=False,
                 mu_excluded=False):

        super(BayesianNN, self).__init__()
        self.mu_excluded = mu_excluded
        self.empirical_complexity_loss_flag = empirical_complexity_loss

        self.weight_prior_sigma = weight_prior_sigma
        self.bias_prior_sigma = bias_prior_sigma

#         self.pi = nn.Parameter(torch.tensor(prior_mix), requires_grad=False)
#         prior_mix = torch.distributions.Categorical(
#             probs=torch.tensor([self.pi, 1 - self.pi])
#         )

#         self.weight_prior_dist = torch.distributions.MixtureSameFamily(
#             prior_mix,
#             torch.distributions.Normal(0,
#                                        torch.tensor([weight_prior_sigma,
#                                                      0.00001])
#                                        )
#         )

#         self.bias_prior_dist = torch.distributions.MixtureSameFamily(
#             prior_mix,
#             torch.distributions.Normal(0,
#                                        torch.tensor([bias_prior_sigma,
#                                                      0.00001])
#                                        )
#         )

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
        x = self.dense_variational_3(x)
        return x

    def empirical_complexity_loss(self):
        empirical_complexity_loss = 0
        for child in self.children():
            empirical_complexity_loss += child.empirical_complexity_loss(
                self.weight_prior_sigma,
                self.bias_prior_sigma
            )
        return empirical_complexity_loss

    def kl_loss(self):
        kl_loss = 0
        for child in self.children():
            kl_loss += child.kl_loss(self.weight_prior_sigma,
                                     self.bias_prior_sigma,
                                     self.mu_excluded)
            # kl_loss += child.kl_loss(self.weight_prior_dist,
            #                          self.bias_prior_dist,
            #                          self.mu_excluded)
        return kl_loss

    def complexity_cost(self):
        if self.empirical_complexity_loss_flag:
            return self.empirical_complexity_loss()
        else:
            return self.kl_loss()
