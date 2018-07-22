from __future__ import print_function, division, absolute_import

import random

import torch
from torch.distributions import constraints
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
from pyro.contrib.gp.kernels import RBF
from pyro.contrib.gp.likelihoods import Gaussian
from pyro.contrib.gp.models import GPModel, SparseGPRegression, GPRegression
from pyro.contrib.gp.util import conditional, Parameterized
# from pyro.params import param_with_module_name
from pyro.optim import Adam

class GP_ADF_RTSS(Parameterized):
    """
    Gaussian process assumed density filter and smoother.
    Reference:
        [1] Analytic Moment-based Gaussian Process Filtering
        [2] A General Perspective on Gaussian Filtering and Smoothing: Explaining Current
            and Deriving New Algorithms
        [3] Robust Filtering and Smoothing with Gaussian Processes
    """

    def __init__(self, input_s, output_s, input_o, output_o, state_dim, observation_dim, option='SSGP', inducing_size=100, name='GP_ADF_RTSS'):
        """
        :param input_s: training inputs for the state transtion model N by D tensor
        :param output_s: training outputs for the state transition model N by E tensor
        :param input_o: training inputs for the observation model N by E tensor
        :param output_o: training outputs for the observation model N by F tensor
        :param state_dim: dimension for the state, D
        :param observation_dim: dimension for the output, E
        :param transition_kernel: kernel function for the
        :param observation_kernel:
        :param options:
        """
        super(GP_ADF_RTSS).__init__(name)
        if option not in ['SSGP', 'GP']:
            raise ValueError('undefined regression option for gp model!')

        assert(input_s.dim() == 2 and output_s.dim() == 2
               and input_o.dim() == 2 and output_o.dim() == 2), "all data inputs can only have 2 dimensions"

        # # use RBF kernel for state transition model and observation model
        # self.state_transition_kernel = RBF(input_dim=state_dim, lengthscale=torch.ones(state_dim) * 0.1)
        # self.observation_kernel = RBF(input_dim=observation_dim, lengthscale=torch.ones(observation_dim) * 0.1)
        self.input_s = input_s
        self.output_s = output_s
        self.input_o = input_o
        self.output_o = output_o

        # choose the model type and initialize based on the option
        self.state_transition_model_list  = []
        self.observation_model_list = []
        if option == 'SSGP':
            for i in range(self.output_s.size()[1]):
                kernel = RBF(input_dim=self.input_s.size()[0], lengthscale=torch.ones(self.input_s.size()[0]))

                range_lis = range(0, output_s.size()[0])
                random.shuffle(range_lis)
                Xu = input_s[input_s[0:inducing_size], :]

                # need to set the name for different model, otherwise pyro will clear the parameter storage
                ssgpmodel = SparseGPRegression(input_s, output_s[:, i], kernel, Xu, name="SSGPs_dim" + str(i))
                self.state_transition_model_list.append(ssgpmodel)

            for i in range(self.output_o.size()[1]):
                kernel = RBF(input_dim=self.input_o.size()[0], lengthscale=torch.ones(self.input_o.size()[0]))

                range_lis = range(0, output_o.size()[0])
                random.shuffle(range_lis)
                Xu = input_o[input_o[0:inducing_size], :]

                ssgpmodel = SparseGPRegression(input_o, output_o[:, i], kernel, Xu, name="SSGPo_dim" + str(i))
                self.state_transition_model_list.append(ssgpmodel)

        else:
            for i in range(self.output_s.size()[1]):
                kernel = RBF(input_dim=self.input_s.size()[0], lengthscale=torch.ones(self.input_s.size()[0]))
                vgpmodel = GPRegression(input_s, output_s[:, i], kernel, name="GPs_dim" + str(i))
                self.observation_model_model_list.append(vgpmodel)

            for i in range(self.output_s.size()[1]):
                kernel = RBF(input_dim=self.input_s.size()[0], lengthscale=torch.ones(self.input_s.size()[0]))
                vgpmodel = GPRegression(input_s, output_s[:, i], kernel, name="GPo_dim"+ str(i))
                self.observation_model_list.append(vgpmodel)

    def fit_GP(self):
        ### train every GPf and GPo, cache necessary variables for further filtering
        self.Kff_s_inv = []
        self.Kff_o_inv = []
        self.K_s_var = []
        self.K_o_var = []
        self.Beta_s = []
        self.Beta_o = []
        self.lengthscale_s = []
        self.lengthscale_o = []


        pyro.clear_param_store()
        num_steps = 2000
        for (i, GPs) in self.state_transition_model_list:
            GPs.optimize(optimizer=Adam({"lr": 0.01}), num_steps=num_steps)
            Kff = GPs.kernel(self.input_s).view(-1)[::self.input_s.size()[0] + 1] + GPs.get_param('noise')
            Lff=  Kff.potrf(upper=False)
            self.Kffs_inv.append(torch.potrs(torch.eye(self.input_s.size()[0]), Lff, upper=False))
            self.Beta_s.append(torch.potrs(self.output_s[:, i], Lff, upper=False))
            self.Ks_var.append(GPs.kernel.get_param("variance"))
            self.lengthscale_s.append(GPs.kernel.get_param("lengthscale"))

        for (i ,GPo) in self.observation_model_list:
            GPo.optimize(optimizer=Adam({"lr": 0.01}), num_steps=num_steps)
            Kff = GPo.kernel(self.input_s).view(-1)[::self.input_s.size()[0] + 1] + GPo.get_param('noise')
            Lff = Kff.potrf(upper=False)
            self.Kffo_inv.append(torch.potrs(torch.eye(self.input_s.size()[0]), Lff, upper=False))
            self.Beta_o.append(torch.potrs(self.output_s[:, i], Lff, upper=False))
            self.Ko_var.append(GPo.kernel.get_param("variance"))
            self.lengthscale_s.append(GPo.kernel.get_param("lengthscale"))


    def mean_propagation(cls, input, Beta, lenthscale, variance, mean, covariance):
        """
        mean of the prpagation of GP for uncertain inputs
        :param input: traing inputs N by D or N by E
        :param Beta: cached Beta N by 1
        :param lengthscale: legnth scale of the RBF kernel  1 by D

        :param variance: variance of the kernel
        :param mean: mean for the uncertain inputs 1 by D or 1 by E
        :param covariance: covariance for the uncertain inputs D by D or E by E
        :return:
        """
        ### porediction of gp mean for uncertain inputs
        assert(input.size()[1] == mean.size()[1])

        # eq 9 of ref. [1]
        with torch.no_grad():

            mat1 = (lenthscale.diag() + covariance)
            det = variance * (torch.det(mat1) ** -0.5) * (torch.det(lenthscale.diag()) ** 0.5)
            diff = input - mean
            # N x 1 x D @ D x D @ N x D x 1 = N x 1 x 1(or D replaced by E) TODO MAYBE CONSIDER ADD SOME JITTER ?
            mat2 = mat1.potrf(upper=False)
            mat3 = torch.potrs(torch.eye(mat1.size()[0]), mat2, upper=False)
            mat4 = (torch.matmul(diff.unsqueeze(1), torch.matmul(mat3, diff.unsqueeze(-1)))) * -0.5
            # (N, )
            l = det * torch.exp(-0.5 * mat4.view(-1))
            mu = torch.matmul(Beta, l)
            return mu

    def variance_propagation(self, input, Beta, lengthscale, Kff_inv, variance, mu, mean, covariance):
        """
        variace of the propagation of GP for uncertain inputs
        :param input: traing inputs N by D or N by E
        :param Beta: cached Beta 1 by N
        :param lengthscale: legnth scale of the RBF kernel  1 by D
        :param Kff_inv: N by N
        :param variance: variance of the kernel
        :param mu: prediction for the mean of GP under uncertain inputs
        :param mean: mean for the uncertain inputs 1 by D or 1 by E
        :param covariance: covariance for the uncertain inputs D by D or E by E
        :return:
        """
        assert (input.size()[1] == mean.size()[1])

        # eq 11 of ref.[1]
        with torch.no_grad():
            mat1 = (lengthscale.diag() / 2. + covariance)
            det =  (torch.det(mat1) ** -0.5) * (torch.det(lengthscale.diag()) ** 0.5)
            # N by 1 by D (E) -/+ N by D (E) = N by N by D (E)
            diff_m = (input.unsqueeze(1) - input) / 2.
            sum_m = (input.unsqueeze(1) + input) / 2.

            mat2 = mat1.potrf(upper=False)
            mat3 = torch.potrs(torch.eye(mat1.size()[0]), mat2, upper=False)

            # elementwise computation
            # N by N
            mat4 = ((diff_m ** 2 / lengthscale * 2).sum(dim=-1)) * -0.5
            # N x N x 1 x D @ D x D @ N x N x D x 1 = N x N x 1 x 1(or D replaced by E) TODO MAYBE CONSIDER ADD SOME JITTER ?
            mat5 = sum_m - mean
            mat6 = (torch.matmul(mat5.unsqueeze(2), torch.matmul(mat3, mat5.unsqueeze(-1)))) * -0.5

            # N by N
            L = variance**2 * det* torch.mul(torch.exp(mat4), torch.exp(mat6.view(input.size()[0], input.size()[0])))
            var = torch.matmul(Beta, torch.matmul(L, Beta)) + variance - torch.trace(torch.matmul(Kff_inv, L)) - mu
            return var

    def covariance_propagation(self, input, Beta_a, lengthscale_a, variance_a, mu_a,
                                            Beta_b, lengthscale_b, variance_b, mu_b,
                                            mean, covariance):
        """

        :param input:  traing inputs N by D or N by E
        :param Beta_a:  cached Beta for output dim a,  1 by N
        :param lengthscale_a: legnth scale of the RBF kernel for output dim a, 1 by D
        :param Kff_inv_a: for output dim a ,N by N
        :param variance_a:  variance of the kernel for output dim a
        :param mu_a: prediction for the mean of GP under uncertain inputs for output dim a
        :param Beta_b: cached Beta for output dim b,  1 by N
        :param lengthscale_b: legnth scale of the RBF kernel for output dim b, 1 by D
        :param Kff_inv_b: for output dim b ,N by N
        :param variance_b: variance of the kernel for output dim b
        :param mu_b: prediction for the mean of GP under uncertain inputs for output dim b
        :param mean: mean for the uncertain inputs 1 by D or 1 by E
        :param covariance: covariance for the uncertain inputs D by D or E by E
        :return:
        """
        assert (input.size()[1] == mean.size()[1])

        # eq 112 of ref.[1]
        with torch.no_grad():

            mat1 = 1 / (1 / lengthscale_a + 1 / lengthscale_b).diag()
            R = mat1 + covariance
            det = (torch.det(R) ** -0.5) * (R ** 0.5)

            # N by 1 by D (E) -/+ N by D (E) = N by N by D (E)
            diff_m = (input.unsqueeze(1) - input) / 2.
            sum_m = (input.unsqueeze(1) * lengthscale_a + input * lengthscale_b) / (lengthscale_a + lengthscale_b)

            mat2 = R.potrf(upper=False)
            mat3 = torch.potrs(torch.eye(mat1.size()[0]), mat2, upper=False)

            # elementwise computation
            # N by N
            mat4 = ((diff_m ** 2 / (lengthscale_a + lengthscale_b)).sum(dim=-1)) * -0.5
            # N x N x 1 x D @ D x D @ N x N x D x 1 = N x N x 1 x 1(or D replaced by E) TODO MAYBE CONSIDER ADD SOME JITTER ?
            mat5 = sum_m - mean
            mat6 = (torch.matmul(mat5.unsqueeze(2), torch.matmul(mat3, mat5.unsqueeze(-1)))) * -0.5
            # N by N
            L = variance_a * variance_b * det * torch.mul(torch.exp(mat4), torch.exp(mat6.view(input.size()[0], input.size()[0])))
            cov = torch.matmul(Beta_a, torch.matmul(L, Beta_b)) - mu_a * mu_b
            return cov











