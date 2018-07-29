from __future__ import print_function, division, absolute_import

import random

import torch
from torch.distributions import constraints
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
from pyro.contrib.gp.kernels import RBF
from pyro.contrib.gp.likelihoods import Gaussian
from pyro.contrib.gp.models import GPModel, SparseGPRegression, GPRegression,VariationalSparseGP
from pyro.contrib.gp.util import conditional, Parameterized
# from pyro.params import param_with_module_name
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

#TODO CACHE THE INVERSE OF LENGTHSCALE ?
class GP_ADF_RTSS(Parameterized):
    """
    Gaussian process assumed density filter and smoother.
    Reference:
        [1] Analytic Moment-based Gaussian Process Filtering
        [2] A General Perspective on Gaussian Filtering and Smoothing: Explaining Current
            and Deriving New Algorithms
        [3] Robust Filtering and Smoothing with Gaussian Processes
    """

    def __init__(self, X_s, y_s, X_o, y_o, option='GP', inducing_size=100, name='GP_ADF_RTSS'):
        """
        :param X_s: training inputs for the state transition model N by D tensor
        :param y_s: training outputs for the state transition model N by E tensor
        :param X_o: training inputs for the observation model N by E tensor
        :param y_o: training outputs for the observation model N by F tensor
        :param state_dim: dimension for the state, D
        :param observation_dim: dimension for the output, E
        :param transition_kernel: kernel function for the
        :param observation_kernel:
        :param options:
        """
        super(GP_ADF_RTSS, self).__init__(name)
        if option not in ['SSGP', 'GP']:
            raise ValueError('undefined regression option for gp model!')

        assert(X_s.dim() == 2 and y_s.dim() == 2
               and X_o.dim() == 2 and y_o.dim() == 2), "all data inputs can only have 2 dimensions"

        # # use RBF kernel for state transition model and observation model
        # self.state_transition_kernel = RBF(input_dim=state_dim, lengthscale=torch.ones(state_dim) * 0.1)
        # self.observation_kernel = RBF(input_dim=observation_dim, lengthscale=torch.ones(observation_dim) * 0.1)
        self.X_s = X_s
        self.y_s = y_s
        self.X_o = X_o
        self.y_o = y_o
        # print(X_s.dtype)
        # print(y_s.dtype)
        # print(X_o.dtype)
        # print(y_o.dtype)

        # choose the model type and initialize based on the option
        self.state_transition_model_list  = []
        self.observation_model_list = []


        if option == 'SSGP':
                for i in range(self.y_s.size()[1]):
                    kernel = RBF(input_dim=self.X_s.size()[1], lengthscale=torch.ones(self.X_s.size()[1]) * 10., variance=torch.tensor(5.0),name="GPs_dim" + str(i) + "_RBF")

                    range_lis = range(0, X_s.size()[0])
                    random.shuffle(range_lis)
                    Xu = X_s[range_lis[0:inducing_size], :]

                    # need to set the name for different model, otherwise pyro will clear the parameter storage
                    ssgpmodel = SparseGPRegression(X_s, y_s[:, i], kernel, Xu, name="SSGPs_model_dim" + str(i), jitter=1e-5)
                    self.state_transition_model_list.append(ssgpmodel)

                for i in range(self.y_o.size()[1]):
                    kernel = RBF(input_dim=self.X_o.size()[1], lengthscale=torch.ones(self.X_o.size()[1]) * 10, variance=torch.tensor(5.0), name="GPo_dim" + str(i) + "_RBF")

                    range_lis = range(0, y_o.size()[0])
                    random.shuffle(range_lis)
                    Xu = X_o[range_lis[0:inducing_size], :]

                    ssgpmodel = SparseGPRegression(X_o, y_o[:, i], kernel, Xu, name="SSGPo_model_dim" + str(i), noise=torch.tensor(2.))
                    self.state_transition_model_list.append(ssgpmodel)

        else:
                for i in range(self.y_s.size()[1]):
                    kernel = RBF(input_dim=self.X_s.size()[1], lengthscale=torch.ones(self.X_s.size()[1]) * 10., variance=torch.tensor(5.0), name="GPs_dim" + str(i) + "_RBF")
                    gpmodel = GPRegression(X_s, y_s[:, i], kernel, name="GPs_model_dim" + str(i), jitter=1e-5)
                    self.state_transition_model_list.append(gpmodel)

                for i in range(self.y_o.size()[1]):
                    kernel = RBF(input_dim=self.X_o.size()[1], lengthscale=torch.ones(self.X_o.size()[1]) * 10., variance=torch.tensor(5.0), name="GPo_dim" + str(i) + "_RBF")
                    gpmodel = GPRegression(X_o, y_o[:, i], kernel, name="GPo_model_dim"+ str(i), noise=torch.tensor(2.))
                    self.observation_model_list.append(gpmodel)

        self.option = option
        #
        # if model_file:
        #     self.load_model(model_file)





        self.mu_s_curr      = torch.zeros(y_s.size()[1])
        self.sigma_s_curr   = torch.eye(y_s.size()[1])
        self.mu_o_curr      = torch.zeros(y_o.size()[1])
        self.sigma_o_curr     = torch.eye(y_s.size()[1])

        self.mu_hat_s_curr      = torch.zeros(y_s.size()[1])
        self.sigma_hat_s_curr   = torch.eye(y_s.size()[1])
        self.mu_hat_s_prev      = torch.zeros(y_s.size()[1])
        self.sigma_hat_s_prev   = torch.eye(y_s.size()[1])

        # For backwards smoothing
        self.mu_hat_s_curr_lis     = []
        self.sigma_hat_s_curr_lis  = []
        self.mu_s_curr_lis         = []
        self.sigma_s_curr_lis      = []

        self.sigma_Xpf_Xcd_lis     = []

        self.Kff_s_inv = torch.zeros((y_s.size()[1], X_s.size()[0], X_s.size()[0]))
        self.Kff_o_inv = torch.zeros((y_o.size()[1], X_o.size()[0], X_o.size()[0]))
        self.K_s_var = torch.zeros(y_s.size()[1], 1)
        self.K_o_var = torch.zeros(y_o.size()[1], 1)
        self.Beta_s = torch.zeros((y_s.size()[1], X_s.size()[0]))
        self.Beta_o = torch.zeros((y_s.size()[1], X_s.size()[0]))
        self.lengthscale_s = torch.zeros((y_s.size()[1], X_s.size()[1]))
        self.lengthscale_o = torch.zeros((y_o.size()[1], X_o.size()[1]))

        if self.option == 'SSGP':
            self.Xu_s = torch.zeros((y_s.size()[1], inducing_size))
            self.Xu_o = torch.zeros((y_o.size()[1], inducing_size))
            self.noise_s = torch.zeros((y_s.size()[1], inducing_size))
            self.noise_o = torch.zeros((y_o.size()[1], inducing_size))

        print("for state transition model, input dim {} and output dim {}".format(X_s.size()[1], y_s.size()[1]))
        print("for observation model, input dim {} and output dim {}".format(X_o.size()[1], y_o.size()[1]))

    def fit_GP(self):
        ### train every GPf and GPo, cache necessary variables for further filtering


        self.GPs_losses = []
        self.GPo_losses = []
        # TODO CACHE DIFFERENT STUFF IF USE SPARSE GP
        pyro.clear_param_store()
        num_steps = 2500
        for (i, GPs) in enumerate(self.state_transition_model_list):
            losses = GPs.optimize(optimizer=Adam({"lr": 0.005}), num_steps=num_steps)
            self.GPs_losses.append(losses)
            print("training for state transition model {} is done!".format(i))



        for (i ,GPo) in enumerate(self.observation_model_list):
            losses = GPo.optimize(optimizer=Adam({"lr": 0.005}), num_steps=num_steps)
            self.GPo_losses.append(losses)
            print("training for observation model {} is done!".format(i))

        self.cache_variable()

        # save the mode
        self._save_model()
        return self.GPs_losses, self.GPo_losses

    def save_model(self):
        pyro.get_param_store().save('gp_adf_rtss.save')

    def load_model(self, filename):
        pyro.get_param_store().load(filename)
        #Beta = self.cache_variable()

    def cache_variable(self):
        #Beta = None
        for (i, GPs) in enumerate(self.state_transition_model_list):
            if self.option == 'GP':
                GPs.guide()
                Kff = GPs.kernel(self.X_s).contiguous()
                Kff.view(-1)[::self.X_s.size()[0] + 1] += GPs.get_param('noise')
                Lff=  Kff.potrf(upper=False)
                self.Kff_s_inv[i, :, :] = torch.potrs(torch.eye(self.X_s.size()[0]), Lff, upper=False)
                self.Beta_s[i, :] = torch.potrs(self.y_s[:, i], Lff, upper=False).squeeze(-1)
                self.K_s_var[i] = GPs.kernel.get_param("variance")
                self.lengthscale_s[i, :] = GPs.kernel.get_param("lengthscale")



            else:
                Xu, noise = GPs.guide()
                if (GPs.approx == 'DTC' or GPs.option == 'VFE'):
                    Kff_inv, Beta = self._compute_cached_var_ssgp(GPs, Xu, noise, "DTC")
                else:
                    Kff_inv, Beta = self._compute_cached_var_ssgp(GPs, Xu, noise, "FITC")

                self.Beta_s[i, :] = Beta
                self.Kff_s_inv[i, :, :] = Kff_inv
                self.K_s_var[i] = GPs.kernel.get_param("variance")
                self.lengthscale_s[i, :] = GPs.kernel.get_param("lengthscale")
                self.Xu_s[i, :] = Xu
                self.noise_s[i, :, :] = noise

        print("variable caching for state transitino model {} is done!".format(i))

        for (i, GPo) in enumerate(self.observation_model_list):
            if self.option== 'GP':
                GPo.guide()
                Kff = GPo.kernel(self.X_o).contiguous()
                Kff.view(-1)[::self.X_o.size()[0] + 1] += GPo.get_param('noise')
                Lff = Kff.potrf(upper=False)
                self.Kff_o_inv[i, :, :] = torch.potrs(torch.eye(self.X_o.size()[0]), Lff, upper=False)
                self.Beta_o[i, :] = torch.potrs(self.y_o[:, i], Lff, upper=False).squeeze(-1)
                self.K_o_var[i] = GPo.kernel.get_param("variance")
                self.lengthscale_o[i, :] = GPo.kernel.get_param("lengthscale")

            else:
                Xu, noise = GPo.guide()
                if (GPo.approx == 'DTC' or GPo.option == 'VFE'):
                    Xu, noise = GPo.guide()
                    Kff_inv, Beta = self._compute_cached_var_ssgp(GPo, Xu, noise, "DTC")
                else:
                    Kff_inv, Beta = self._compute_cached_var_ssgp(GPo, Xu, noise, "FITC")

                self.Beta_o[i, :] = Beta
                self.Kff_o_inv[i, :, :] = Kff_inv
                self.K_o_var[i] = GPo.kernel.get_param("variance")
                self.lengthscale_o[i, :] = GPo.kernel.get_param("lengthscale")
                self.Xu_o[i, :] = Xu
                self.noise_o[i, :, :] = noise
            print("variable caching for observation model {} is done!".format(i))

            # self.zip_cached_s = list(zip(self.Beta_s, self.lengthscale_s, self.K_s_var, self.Kff_s_inv))
            # self.zip_cached_o = list(zip(self.Beta_o, self.lengthscale_o, self.K_o_var, self.Kff_o_inv))
            print(self.Beta_s.size())
            print(self.Beta_o.size())
            print(self.lengthscale_s.size())
            print(self.lengthscale_o.size())
            print(self.Kff_s_inv.size())
            print(self.Kff_o_inv.size())
            print(self.K_s_var.size())
            print(self.K_o_var.size())

            print("initialization is done!")

    def _compute_cached_var_ssgp(self, model, Xu, noise, option):


        M = Xu.size()[0]
        if (option == 'DTC' or option == 'VFE'):
            Kuu = model.kernel(Xu).contiguous()
            Kuu.view(-1)[::M + 1] += model.jitter  # add jitter to the diagonal
            Luu = Kuu.potrf(upper=False)

            Kuf = model.kernel(Xu, model.X)
            # variance = model.kernel.get_param("variance")

            Epi = torch.matmul(Kuf, Kuf.t()) / noise + Kuu
            L_Epi = Epi.potrf(upper=False)
            Kff_inv = torch.potrs(torch.eye(Xu.size()[0]), L_Epi) - torch.potrs(torch.eye(Xu.size()[0]), Luu)
            Beta = torch.potrs(torch.matmul(Kuf, model.y), L_Epi) / noise

        else:
            Kuu = model.kernel(Xu).contiguous()
            Kuu.view(-1)[::M + 1] += model.jitter  # add jitter to the diagonal
            Luu = Kuu.potrf(upper=False)

            Kuf = model.kernel(Xu, model.X)

            W = Kuf.trtrs(Luu, upper=False)[0].t()
            Delta = model.kernel(model.X, diag=True) - W.pow(2).sum(dim=-1) + noise.expand(W.shape[0])
            L_Delta = Delta.potrf(upper=False)

            U = Kuf.t().trtrs(L_Delta, upper=False)[0]

            Epi = Kuu + torch.matmul(U.t(), U)
            L_Epi = Epi.potrf(upper=False)

            Z = torch.matmul(U.t(), model.y.trtrs(L_Delta))
            Beta = torch.potrs(Z, L_Epi)

            Kff_inv = torch.potrs(torch.eye(Xu.size()[0]), L_Epi) + torch.potrs(torch.eye(Xu.size()[0]), Luu)

        return Kff_inv, Beta




    def mean_propagation(self, input, Beta, lengthscale, variance, mean, covariance):
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
        # print(input.size())
        # print(Beta.size())
        # print(lengthscale.size())
        # print(variance.size())
        # print(mean.size())
        # print(covariance.size())

        assert(input.size()[1] == mean.size()[1])

        # eq 9 of ref. [1]
        with torch.no_grad():

            mat1 = (lengthscale.diag() + covariance)

            det = variance * (torch.det(mat1) ** -0.5) * (torch.det(lengthscale.diag()) ** 0.5)
            diff = input - mean
            # N x 1 x D @ D x D @ N x D x 1 = N x 1 x 1(or D replaced by E) TODO MAYBE CONSIDER ADD SOME JITTER ?
            mat2 = mat1.potrf(upper=False)
            mat3 = torch.potrs(torch.eye(mat1.size()[0]), mat2, upper=False)
            mat4 = (torch.matmul(diff.unsqueeze(1), torch.matmul(mat3, diff.unsqueeze(-1)))) * -0.5
            # (N, )
            l = det * torch.exp(mat4.view(-1))
            mu = torch.matmul(Beta, l)

            return mu

    def variance_propagation(self, input, Beta, lengthscale, variance, Kff_inv, mu, mean, covariance):
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

            # print(mat3.size(), mat5.size())
            mat6 = (torch.matmul(mat5.unsqueeze(2), torch.matmul(mat3, mat5.unsqueeze(-1)))) * -0.5
            # N by N
            L = variance**2 * det* torch.mul(torch.exp(mat4), torch.exp(mat6.view(input.size()[0], input.size()[0])))
            var = torch.matmul(Beta, torch.matmul(L, Beta)) + variance - torch.trace(torch.matmul(Kff_inv, L)) - mu * mu
            return var.diag()

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

        # eq 12 of ref.[1]
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

    def _prediction(self, input, Beta, lengthscale, var, Kff_inv, mean, covariance):
        """
        prediction from p(x(k-1) | y(1:k-1) to p(x(k) | y(1:k-1)), p(x(k-1) | y(1:k-1)) is the filtered result of the last step
            OR
        prediction from p(x(k) | y(1:k-1) to p(y(k) | y(1:k-1)), p(x(k) | y(1:k-1)) is the predicted result from this step
        :param mean: mean vector for p(x(k-1) | y(1:k-1) 1 by D
        :param covariance: covariance matrix for p(x(k-1) | y(1:k-1)
        :return:
        """
        range_lis = [i for i in range(Beta.size()[0])]
        pred_mean_tensor = torch.tensor(list(map(lambda i : self.mean_propagation(input, Beta[i, :], lengthscale[i, :], var[i, :], mean, covariance), range_lis)))
        pred_cov_diag = torch.tensor(list(map(lambda i : self.variance_propagation(input, Beta[i, :], lengthscale[i, :], var[i, :], Kff_inv[i, :, :],
                                                                                   pred_mean_tensor, mean, covariance), range_lis)))


        pred_cov_diag = pred_cov_diag / 2.
        #pred_cov_diag = torch.eye(1)
        if Beta.size()[0] > 1:
            range_lis = [(i ,j) for i in range(0, Beta.size()[0]) for j in range(i, Beta.size()[0])]
            list_cov = list(map(lambda tup : self.covariance_propagation(input, Beta[tup[0], :], lengthscale[tup[0], :], var[tup[0], :], pred_mean_tensor[tup[0]],
                                                                                Beta[tup[1], :], lengthscale[tup[1], :], var[tup[1], :], pred_mean_tensor[tup[1]],
                                                                                mean, covariance), range_lis))

        pred_cov_diag = torch.tensor(list_cov).view(pred_mean_tensor.size()[0], -1)

        pred_cov = torch.ones((Beta.size()[0], Beta.size()[0])) - torch.eye((Beta.size([0], Beta.size()[0])))
        pred_cov[torch.triu(torch.ones(Beta.size()[0], Beta.size()[0])) == 1] = pred_cov
        pred_cov = torch.mul(pred_cov, torch.eye((Beta.size()[0], Beta.size()[0]) * 0.5))

        pred_cov = pred_cov + pred_cov.transpose(dim0=0, dim1=1)
        # pred_covariance_tensor = pred_covariance_tensor + pred_covariance_tensor.transpose(dim0=0, dim1=1)
        #
        return pred_mean_tensor, pred_cov_diag

    def step(self):
        """
        set curret filtered value to previous filtered value when new observation arrives
        :return:
        """
        self.mu_hat_s_prev = self.mu_hat_s_curr.clone()
        self.sigma_hat_s_prev = self.sigma_hat_s_curr.clone()

    def prediction(self, mu_hat_s_prev, sigma_hat_s_prev, index=None):

        assert(mu_hat_s_prev.dim() == 2),"filtered mean of previous step needs to have dim 2, has {} instead.".format(mu_hat_s_prev.dim())
        assert (sigma_hat_s_prev.dim() == 2), "filtered covariance of previous step needs to have dim 2, has {} instead.".format(sigma_hat_s_prev.dim())

        #mean_predicted_s_curr, covariance_predicted_s_curr = self._prediction(self.X_s, self.zip_cached_s, mu_hat_s_prev, sigma_hat_s_prev)
        if self.option == "GP":
            mu_s_curr, sigma_s_curr = self._prediction(self.X_s, self.Beta_s, self.lengthscale_s, self.K_s_var, self.Kff_s_inv, mu_hat_s_prev, sigma_hat_s_prev)
            #sigma_Xcd_Xpf, sigma_Xpf_Xcd = self._compute_cov(self.X_s, mu_hat_s_prev, self.mu_s_curr,
            #                                                 self.lengthscale_s, sigma_hat_s_prev, self.K_s_var,
            #                                                 self.Beta_s)
        else:
            assert(index != 0 and index <= len(self.Xu_s)), "state transition models have dimension {}, index is {}.".format(len(self.Xu_s), index)
            mu_s_curr, sigma_s_curr = self._prediction(self.Xu_s[index], self.zip_cached_s, mu_hat_s_prev, sigma_hat_s_prev)
        #     sigma_Xcd_Xpf, sigma_Xpf_Xcd = self._compute_cov(self.Xu_s[index], mu_hat_s_prev, self.mu_s_curr,
        #                                                      self.lengthscale_s, sigma_hat_s_prev, self.K_s_var,
        #                                                      self.Beta_s)
        #
        self.mu_s_curr, self.sigma_s_curr = mu_s_curr, sigma_s_curr
        self.mu_s_curr_lis.append(self.mu_s_curr.clone())
        self.sigma_s_curr_lis.append(self.sigma_s_curr.clone())
        #
        #
        # self.sigma_Xpf_Xcd_lis.append(sigma_Xpf_Xcd .clone())
        #
        return self.mu_s_curr, self.sigma_s_curr

    def filtering(self, observation, mu_s_curr, sigma_s_curr, index=None):
        """
        filtering from p(x(k) | y(1:k-1)), updated using p(y(k) | x(k)), to get p(x(k) | y(1:k))
        :param mean_pred: mean of p(x(k) | y(1:k-1)),
        :param covariance_pred: covariance of p(x(k) | y(1:k-1))
        :return:
        """

        # first compute the predtion of measurement based on the observation model
        if self.option == "GP":
            mu_o_curr, sigma_o_curr = self._prediction(self.X_o, self.zip_cached_o, mu_s_curr, sigma_s_curr)
            Cov_yx, Cov_xy = self._compute_cov(self.X_o, mu_s_curr, self.mu_o_curr,
                                               self.lengthscale_o, sigma_s_curr, self.K_o_var, self.Beta_o)
        else:
            assert (index != 0 and index <= len(self.Xu_o)), "state transition models have dimension {}, index is {}.".format(len(self.Xu_o), index)
            mu_s_curr, sigma_s_curr = self._prediction(self.Xu_o[index], self.zip_cached_o, mu_s_curr, sigma_s_curr)
            Cov_yx, Cov_xy = self._compute_cov(self.Xu_o[index], mu_s_curr, self.mu_o_curr,
                                               self.lengthscale_o, sigma_s_curr, self.K_o_var, self.Beta_o)
            
        self.mu_o_curr, self.sigma_o_curr = mu_o_curr, sigma_o_curr

        sigma_o_curr_inv = torch.potrs(sigma_o_curr.potrf(upper=False), torch.eye(sigma_o_curr.size()[0]), upper=False)
        mu_hat_s_curr = mu_s_curr + torch.matmul(Cov_xy, torch.matmul(sigma_o_curr_inv, (observation - mu_o_curr)))
        sigma_hat_s_curr = sigma_s_curr - torch.matmul(Cov_xy, torch.matmul(sigma_o_curr_inv, Cov_yx))

        self.mu_hat_s_curr, self.sigma_hat_s_curr = mu_hat_s_curr, sigma_hat_s_curr
        self.mu_hat_s_curr_lis.append(self.mu_hat_s_curr.clone())
        self.sigma_hat_s_curr_lis.append(self.sigma_hat_s_curr.clone())


        return mu_hat_s_curr, sigma_hat_s_curr

    def smoothing(self, steps=10):
        """
        Perform smoothing from the most recent step
        :param steps: how many steps for the backward smoothing
        :return:
        """

        mu_s_curr_lis = self.mu_s_curr_lis[::-1]
        sigma_s_curr_lis = self.sigma_s_curr_lis[::-1]
        mu_hat_s_curr_lis = self.mu_hat_s_curr_lis[::-1]
        sigma_hat_s_curr_lis = self.sigma_hat_s_curr_lis[::-1]
        sigma_Xpf_Xcd_lis = self.sigma_Xpf_Xcd_lis[::-1]

        # start from the most recent step
        mu_smoothed_curr = mu_hat_s_curr_lis[0]
        sigma_smoothed_curr = sigma_hat_s_curr_lis[0]

        mu_smoothed_curr_lis = [mu_smoothed_curr.clone()]
        sigma_smoothed_curr_lis = [sigma_smoothed_curr.clone()]


        for i in range(0, steps):
            # first compute the joint distribution of p(x(t-1), x(t) | z(1:t-1))
            sigma_Xpf_Xcd     = sigma_Xpf_Xcd_lis[i]

            mu_hat_s          = mu_hat_s_curr_lis[i+1]
            sigma_hat_s       = sigma_hat_s_curr_lis[i+1]

            mu_s              = mu_s_curr_lis[1]
            sigma_s           = sigma_s_curr_lis[i]

            Mat1 = torch.potrs(sigma_s.potrf(upper=False), torch.eye(sigma_hat_s.size()[0]), upper=False)
            J_prev = torch.matmul(sigma_Xpf_Xcd, Mat1)

            mu_smoothed_curr = mu_hat_s + torch.matmul(J_prev, (mu_smoothed_curr - mu_s))
            sigma_smoothed_curr = sigma_hat_s + torch.matmul(J_prev, torch.matmul(sigma_smoothed_curr - sigma_s, J_prev))

            mu_smoothed_curr_lis.append(mu_smoothed_curr.clone())
            sigma_smoothed_curr_lis.append(sigma_smoothed_curr.clone())

        return mu_smoothed_curr_lis, sigma_smoothed_curr_lis


    def _compute_cov(self, mu1, mu2, mu3, lengthscale, cov, var, Beta):
        # N x D
        mu1 = mu1
        # 1 x D
        mu2 = mu2
        # a list of D X D, length E
        cov_1 = lengthscale
        # D x D
        cov_2 = cov
        # E x 1
        var = var

        # N x E x 1
        Beta = Beta

        # E x D x D tensor
        Mat1 = list(map(lambda x: x.diag(), lengthscale))
        # E x D x D tensor
        Mat2 = list(map(lambda x: torch.potrs((x + cov_2).potrf(upper=False), torch.eye(x.size()), upper=False), Mat1))
        # Mat3 = torch.stack(Mat1) + torch.matmul(torch.stack(Mat1), torch.matmul(Mat2, torch.stack(Mat1)))
        # Mat4 = cov_2 + torch.matmul(cov_2, torch.matmul(Mat2, cov_2))
        # N x E x D x 1
        Mu = torch.stack(mu1).unsqueeze(1).unsqueeze(-1) \
             - torch.matmul(torch.stack(Mat1),
                            torch.matmul(torch.stack(Mat2), torch.stack(mu1).unsqueeze(1).unsqueeze(-1))) \
             + mu2.unsqueeze(1).unsqueeze(-1) - torch.matmul(cov_2, torch.matmul(torch.stack(Mat2),
                                                                                 mu2.unsqueeze(1).unsqueeze(-1)))
        # N x E x D
        Mu = Mu.squeeze(-1)

        #### TODO change it to a lambda func ?
        range_lis = range(0, mu1.size()[1])
        Det1_func = list(map(lambda i: torch.det(torch.stack(cov_1)[i, :, :]), range_lis))
        Det2_func = list(map(lambda i: torch.det((torch.stack(cov_1) + cov_2)[i, :, :]), range_lis))

        # E
        Det = torch.mul(torch.stack(Det1_func) ** 0.5 * torch.stack(Det2_func) ** -0.5, torch.tensor(var))
        ####

        # N x E x 1 x 1
        Mat3 = torch.matmul((torch.stack(mu1) - mu2).unsqueeze(1).unsqueeze(1),
                            torch.matmul(torch.stack(Mat2), (torch.stack(mu1) - mu2).unsqueeze(1).unsqueeze(-1)))
        Z = torch.mul(Det, torch.exp(-0.5 * Mat3))

        # N x E x D
        Cov_yx = torch.matmul(Beta, torch.mul(Z.squeeze(-1), Mu))
        # E x D
        Cov = torch.sum(Cov_yx, dim=0) - torch.ger(mu3, mu2)
        Cov_T = Cov_yx.transpose(dim0=0, dim1=1)

        return Cov, Cov_T


if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt
    import random

    plt.style.use("seaborn")
    pyro.set_rng_seed(0)


    def ONEDexample(x, noise = True):
        x_next = 0.5 * x + 25 * x / (1 + x ** 2)
        if noise:
             x_next += np.random.normal(scale=0.2)
        y = 5 * np.sin(x * 2 )
        if noise:
            y += np.random.normal(scale=0.01)

        return x_next, y

    input = np.linspace(-10, 10, 1000)
    input_next = []
    for x in input:
        input_next.append(ONEDexample(x)[0])


    # plt.scatter(input, np.array(input_next), marker='o', color='b', label='true', s=10)
    # plt.xlabel(r'$\mu_0$')
    # plt.legend()
    # plt.show()

    # Generate the training set

    index = list(range(0, 1000))
    random.shuffle(index)
    # X_s = input[index[0:200]]
    # X_o = X_s.copy()
    # y_s = []
    # y_o = []
    # for x in X_s:
    #     y_s.append(ONEDexample(x)[0])
    #     y_o.append(ONEDexample(x, noise=True)[1])
    N=200
    X_s = dist.Uniform(-10., 10.0).sample(sample_shape=(N,))
    X_o = X_s.clone()
    y_s = 0.5 * X_s + 25 * X_s / (1 + X_s ** 2) + dist.Normal(0.0, 0.2).sample(sample_shape=(N,))
    y_o = 5 * torch.sin(2 * X_s) + dist.Normal(0.0, 0.01).sample(sample_shape=(N,))


    X_s = torch.tensor(X_s, dtype=torch.float32).unsqueeze(-1)
    y_s = torch.tensor(y_s, dtype=torch.float32).unsqueeze(-1)
    X_o = torch.tensor(X_o, dtype=torch.float32).unsqueeze(-1)
    y_o = torch.tensor(y_o, dtype=torch.float32).unsqueeze(-1)

    # print(X_o)
    # print(y_o)

    # plt.figure(0)
    #
    # # plt.subplot(211)
    # plt.scatter(X_s.detach().numpy(), y_s.detach().numpy(), marker='o', color='b', label='true_state', s=10)
    # plt.xlabel(r'$\mu_0$')
    # plt.legend()
    # #plt.show()
    #
    # # plt.subplot(212)
    # plt.scatter(X_s.detach().numpy(), y_o.detach().numpy(), marker='*', color='r', label='observation', s=10)
    # plt.xlabel(r'$\mu_0$')
    # plt.legend()
    #
    # plt.show()
    # print(X_s.size())
    # print(y_s.size())
    # print(X_o.size())
    # print(y_o.size())
    pyro.get_param_store().load('gp_adf_rtss.save')
    gp_adf_rtss = GP_ADF_RTSS(X_s, y_s, X_o, y_o, option='GP')
    pyro.module('GP_ADF_RTSS', gp_adf_rtss, update_module_params=True)
    #gps_losses, gpo_losses = gp_adf_rtss.fit_GP()

    # plt.subplot(211)
    # plt.plot(gps_losses[0])
    # plt.subplot(212)
    # plt.plot(gpo_losses[0])
    # plt.show()
    # print(len(gp_adf_rtss.state_transition_model_list))
    # print(len(gp_adf_rtss.observation_model_list))
    #
    # print(len(gp_adf_rtss.state_transition_model_list[-1].kernel.get_param("lengthscale")))
    # print(len(gp_adf_rtss.observation_model_list[-1].kernel.get_param("lengthscale")))


    def plot(X, y, plot_observed_data=False, plot_predictions=False, n_prior_samples=0,
             model=None, kernel=None, n_test=500):

        plt.figure(figsize=(12, 6))
        if plot_observed_data:
            plt.plot(X.numpy(), y.numpy(), 'kx')
        if plot_predictions:
            Xtest = torch.linspace(-10, 10.0, n_test)  # test inputs
            # compute predictive mean and variance
            with torch.no_grad():
                if type(model) == VariationalSparseGP:
                    mean, cov = model(Xtest, full_cov=True)
                else:
                    mean, cov = model(Xtest.unsqueeze(-1), full_cov=True, noiseless=False)
            sd = cov.diag().sqrt()  # standard deviation at each input point x
            plt.plot(Xtest.numpy(), mean.numpy(), 'r', lw=2)  # plot the mean
            plt.fill_between(Xtest.numpy(),  # plot the two-sigma uncertainty about the mean
                             (mean - 2.0 * sd).numpy(),
                             (mean + 2.0 * sd).numpy(),
                             color='C0', alpha=0.3)
        if n_prior_samples > 0:  # plot samples from the GP prior
            Xtest = torch.linspace(-10.0, 10.0, n_test) # test inputs
            noise = (model.noise if type(model) != VariationalSparseGP
                     else model.likelihood.variance)
            cov = kernel.forward(Xtest.unsqueeze(-1)) + noise.expand(n_test).diag()
            samples = dist.MultivariateNormal(torch.zeros(n_test), covariance_matrix=cov) \
                .sample(sample_shape=(n_prior_samples,))
            plt.plot(Xtest.numpy(), samples.numpy().T, lw=2, alpha=0.4)

        plt.xlim(-10.0, 10.0)
        plt.show()


    # kernel = RBF(input_dim=1,  lengthscale=torch.tensor(10.), variance=torch.tensor(5.))
    # gpr = GPRegression(X_o[:, 0], y_o[:, 0], kernel, noise=torch.tensor(1.))
    #
    # optim = Adam({"lr": 0.005})
    # svi = SVI(gpr.model, gpr.guide, optim, loss=Trace_ELBO())
    # losses = []
    # num_steps = 2500
    # for i in range(num_steps):
    #     losses.append(svi.step())
    # plt.plot(losses)
    # plt.show()
    # # #

    #gp_adf_rtss.load_model()

    ssmodel = gp_adf_rtss.state_transition_model_list[-1]
    obmodel = gp_adf_rtss.observation_model_list[-1]
    # plot(ssmodel.X[:, 0], ssmodel.y, model=ssmodel, plot_observed_data=True, plot_predictions=True)
    # plot(obmodel.X[:, 0], obmodel.y, model=obmodel, plot_observed_data=True, plot_predictions=True)

    # Draw the 200 independant pairs


    N = 500
    X = dist.Uniform(-10., 10.0).sample(sample_shape=(N,))
    sigma = torch.tensor(0.25)
    X_next =  0.5 * X + 25 * X / (1 + X ** 2) #+ dist.Normal(0.0, 0.2).sample(sample_shape=(N,))
    y_observe =  5 * torch.sin(2 * X_next) + dist.Normal(0.0, 0.01).sample(sample_shape=(N,))

    zipped_input = list(zip(X, y_observe))

    gp_adf_rtss.cache_variable()
    for (i, input) in enumerate(zipped_input):


        mu_pred, sigma_pred = gp_adf_rtss.prediction(input[0].unsqueeze(-1).unsqueeze(-1), sigma.unsqueeze(-1).unsqueeze(-1))

        print(X_next[i], mu_pred, sigma_pred)

    # print(gp_adf_rtss.state_transition_model_list[0].kernel.get_param("lengthscale"), beta)
    #
    # #
    # #plot(plot_observed_data=True)  # let's
    #

    # Kff = gp_adf_rtss.state_transition_model_list[0].kernel(gp_adf_rtss.state_transition_model_list[0].X).contiguous()
    # Kff.view(-1)[::200 + 1] += gp_adf_rtss.state_transition_model_list[0].get_param('noise')  # add noise to the diagonal
    # Lff = Kff.potrf(upper=False)
    #
    # print(Lff_s[0:3, 0:3], Lff[0:3, 0:3])






























