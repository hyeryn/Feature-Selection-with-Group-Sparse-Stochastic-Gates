import torch

class PoiBin(object):

    def __init__(self, probabilities, args):
        self.args = args
        self.success_probabilities = probabilities    # each prob (0~1)
        self.number_trials = self.success_probabilities.shape[0]
        self.omega = 2 * torch.pi / (self.number_trials + 1)
        self.pmf_list = self.get_pmf_xi()
        self.cdf_0 = self.pmf_list[0]
        self.cdf_list = self.get_cdf(self.pmf_list)

    # number = integer and not negative (p=1이 몇개인가)
    def pmf(self, number_successes):
        return self.pmf_list[number_successes]

    def cdf(self, number_successes):
        return self.cdf_list[number_successes]

    def get_cdf(self, event_probabilities):
        cdf = torch.empty(self.number_trials + 1)
        cdf[0] = event_probabilities[0]
        for i in range(1, self.number_trials + 1):
            cdf[i] = cdf[i - 1] + event_probabilities[i]
        return cdf

    def get_pmf_xi(self):               
        # `\\xi(k) = pmf(k) = Pr(X = k)`
        chi_real = torch.empty(self.number_trials + 1)
        chi_imag = torch.empty(self.number_trials + 1)
        chi = torch.complex(chi_real, chi_imag).to(self.args.device)
        chi[0] = 1
        half_number_trials = int(
            self.number_trials / 2 + self.number_trials % 2)
        # set first half of chis:
        chi[1:half_number_trials + 1] = self.get_chi(
            torch.arange(1, half_number_trials + 1))
        # set second half of chis:      # 이전 matrix 순서 역의 켤레
        chi[half_number_trials + 1:self.number_trials + 1] = torch.conj(
            torch.flip(chi[1:self.number_trials - half_number_trials + 1] , [0]))
        chi /= self.number_trials + 1
        xi = torch.fft.fft(chi)

        # 일단 real값만 바로 받아오기. 
        xi = xi.real
        xi += torch.finfo(float).eps     # noise
        return xi

    def get_chi(self, idx_array):
        # get_z:
        exp_value = torch.exp(self.omega * idx_array * 1j).to(self.args.device)
        xy = 1 - self.success_probabilities + \
            self.success_probabilities * exp_value.unsqueeze(1)
        # sum over the principal values of the arguments of z:
        argz_sum = torch.arctan2(xy.imag, xy.real).sum(axis=1)
        # get d value:
        exparg = torch.log(torch.abs(xy)).sum(axis=1)
        d_value = torch.exp(exparg)
        # get chi values:
        chi = d_value * torch.exp(argz_sum * 1j)
        return chi
