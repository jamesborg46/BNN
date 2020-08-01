import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import batch_linear

import math


class DenseVariational(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 explicit_gradient_flag=False,
                 bias=True):

        super(DenseVariational, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.explicit_gradient_flag = explicit_gradient_flag

        self.weight_mu = nn.Parameter(
            torch.Tensor(out_features, in_features))

        self.weight_rho = nn.Parameter(
            torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features)

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
        self.bias_sigma = torch.log(1 + torch.exp(self.bias_rho))

        self.weight_dist = (torch.distributions
                                 .Normal(self.weight_mu, self.weight_sigma))

        self.bias_dist = (torch.distributions
                               .Normal(self.bias_mu, self.bias_sigma))

        if self.explicit_gradient_flag:
            self.bias = self.bias_dist.sample((samples,)).requires_grad_()
            self.weight = self.weight_dist.sample((samples,)).requires_grad_()
        else:
            self.bias = self.bias_dist.rsample((samples,))
            self.weight = self.weight_dist.rsample((samples,))

        return batch_linear(input, self.weight, self.bias)

    def empirical_complexity_loss(self, weight_prior_dist, bias_prior_dist):
        weight_log_prob = (torch.distributions
                                .Normal(self.weight_mu, self.weight_sigma)
                                .log_prob(self.weight))

        bias_log_prob = (torch.distributions
                              .Normal(self.bias_mu, self.bias_sigma)
                              .log_prob(self.bias))

        weight_prior_log_prob = weight_prior_dist.log_prob(self.weight)
        bias_prior_log_prob = bias_prior_dist.log_prob(self.bias)

        empirical_complexity_loss = (
            torch.sum(weight_log_prob - weight_prior_log_prob, dim=[1,2]) +
            torch.sum(bias_log_prob - bias_prior_log_prob, dim=1)
        )

        return empirical_complexity_loss

    def analytical_complexity_loss(self, weight_prior_dist, bias_prior_dist):
        weight_mu = self.weight_mu
        bias_mu = self.bias_mu

        weights_kl_loss = torch.sum(
            torch.distributions.kl_divergence(self.weight_dist,
                                              weight_prior_dist)
        )

        bias_kl_loss = torch.sum(
            torch.distributions.kl_divergence(self.bias_dist,
                                              bias_prior_dist)
        )

        return weights_kl_loss + bias_kl_loss

    def explicit_gradient_calc(self, sampled_losses):
        assert sampled_losses.shape[0] == self.weight.shape[0]
        num_samples = sampled_losses.shape[0]

        weight_mu_grads = []
        weight_rho_grads = []
        bias_mu_grads = []
        bias_rho_grads = []

        for i in range(num_samples):
            loss = sampled_losses[i]

            dl_dw = torch.autograd.grad(loss,
                                        self.weight,
                                        retain_graph=True)[0][i]

            dl_dwmu = torch.autograd.grad(loss,
                                          self.weight_mu,
                                          retain_graph=True)[0]

            dl_dwrho = torch.autograd.grad(loss,
                                           self.weight_rho,
                                           retain_graph=True)[0]

            weight_eps = (self.weight[i] - self.weight_mu) / self.weight_sigma

            weight_mu_grads.append(dl_dw + dl_dwmu)
            weight_rho_grads.append(
                dl_dw * (weight_eps / (1 + torch.exp(-self.weight_rho)))
                + dl_dwrho
            )

            dl_db = torch.autograd.grad(loss,
                                        self.bias,
                                        retain_graph=True)[0][i]

            dl_dbmu = torch.autograd.grad(loss,
                                          self.bias_mu,
                                          retain_graph=True)[0]

            dl_dbrho = torch.autograd.grad(loss,
                                           self.bias_rho,
                                           retain_graph=True)[0]

            bias_eps = (self.bias[i] - self.bias_mu) / self.bias_sigma

            bias_mu_grads.append(dl_db + dl_dbmu)
            bias_rho_grads.append(
                dl_db * (bias_eps / (1 + torch.exp(-self.bias_rho)))
                + dl_dbrho
            )

        self.weight_mu.grad = torch.mean(torch.stack(weight_mu_grads),
                                         dim=0)
        self.weight_rho.grad = torch.mean(torch.stack(weight_rho_grads),
                                                      dim=0)
        self.bias_mu.grad = torch.mean(torch.stack(bias_mu_grads),
                                       dim=0)
        self.bias_rho.grad = torch.mean(torch.stack(bias_rho_grads),
                                        dim=0)


class BayesianNN(nn.Module):

    def __init__(self,
                 weight_prior_dist,
                 bias_prior_dist,
                 activation_function=None,
                 prior_mix=1,
                 empirical_complexity_loss=False,
                 explicit_gradient=False):

        super(BayesianNN, self).__init__()
        self.empirical_complexity_loss_flag = empirical_complexity_loss
        self.explicit_gradient_flag = explicit_gradient

        # self.weight_prior_dist = weight_prior_dist
        # self.bias_prior_dist = bias_prior_dist

        self.pi = nn.Parameter(torch.tensor(prior_mix), requires_grad=False).to('cuda:0')
        prior_mix = torch.distributions.Categorical(
            probs=torch.tensor([self.pi, 1 - self.pi]).to('cuda:0')
        )

        self.weight_prior_dist = torch.distributions.MixtureSameFamily(
            prior_mix,
            torch.distributions.Normal(0,
                                       torch.tensor([0.1,
                                                     0.00001]).to('cuda:0')
                                       )
        )

        self.bias_prior_dist = torch.distributions.MixtureSameFamily(
            prior_mix,
            torch.distributions.Normal(0,
                                       torch.tensor([1,
                                                     0.00001]).to('cuda:0')
                                       )
        )

        self.dense_variational_1 = DenseVariational(784,
                                                    1200,
                                                    explicit_gradient)

        self.dense_variational_2 = DenseVariational(1200,
                                                    1200,
                                                    explicit_gradient)

        self.dense_variational_3 = DenseVariational(1200,
                                                    10,
                                                    explicit_gradient)

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
                self.weight_prior_dist,
                self.bias_prior_dist
            )
        return empirical_complexity_loss

    def analytical_complexity_loss(self):
        analytical_complexity_loss = 0
        for child in self.children():
            analytical_complexity_loss += child.analytical_complexity_loss(
                self.weight_prior_dist,
                self.bias_prior_dist
            )
        return analytical_complexity_loss

    def complexity_cost(self):
        if self.empirical_complexity_loss_flag:
            return self.empirical_complexity_loss()
        else:
            return self.analytical_complexity_loss()

    def explicit_gradient_calc(self, sampled_losses):
        for child in self.children():
            child.explicit_gradient_calc(sampled_losses)

    def propagate_loss(self, sampled_losses):
        if self.explicit_gradient_flag:
            self.explicit_gradient_calc(sampled_losses)
        else:
            loss = torch.mean(sampled_losses)
            loss.backward()

