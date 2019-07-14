import torch
import torch.nn.functional as F

class MSE(torch.nn.Module):

    def __init__(self, model):
        super(MSE, self).__init__()

        self.model = model
        self.hidden_features = torch.nn.Sequential(
            *list(model.children())[:]
        )

    def forward(self, x):
        return self.model(x)

    def forward_da(self, a, b, c):
        a = self.hidden_features(a)
        b = self.hidden_features(b)
        c = self.hidden_features(c)

        return (
                       self.mse(a, b) +
                       self.mse(b, c) +
                       self.mse(c, a)
               ) / 3

    @staticmethod
    def mse(x, y):
        return torch.mean(torch.pow(x - y, 2))


class MMD(torch.nn.Module):

    def __init__(self, model, scales=[0.2, 0.5, 0.9, 1.3], base=1.0, biased=False):
        super(MMD, self).__init__()

        self.model = model
        self.hidden_features = torch.nn.Sequential(
            *list(model.children())[:]
        )

        self.scales = scales
        self.base = base
        self.biased = biased

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def _mix_rbf_kernel(X, Y, sigma_list):
        assert (X.size(0) == Y.size(0))
        m = X.size(0)

        Z = torch.cat((X, Y), 0)
        ZZT = torch.mm(Z, Z.t())
        diag_ZZT = torch.diag(ZZT).unsqueeze(1)
        Z_norm_sqr = diag_ZZT.expand_as(ZZT)
        exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

        K = 0.0
        d = len(sigma_list)
        for sigma in sigma_list:
            gamma = 1.0 / (2 * sigma ** 2)
            K += torch.exp(-gamma * exponent) * (1 / d)

        return K[:m, :m], K[:m, m:], K[m:, m:], d

    @staticmethod
    def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
        m = K_XX.size(0)  # assume X, Y are same shape

        # Get the various sums of kernels that we'll use
        # Kts drop the diagonal, but we don't need to compute them explicitly
        if const_diagonal is not False:
            diag_X = diag_Y = const_diagonal
            sum_diag_X = sum_diag_Y = m * const_diagonal
        else:
            diag_X = torch.diag(K_XX)  # (m,)
            diag_Y = torch.diag(K_YY)  # (m,)
            sum_diag_X = torch.sum(diag_X)
            sum_diag_Y = torch.sum(diag_Y)

        Kt_XX_sums = K_XX.sum(dim=1) - diag_X  # \tilde{K}_XX * e = K_XX * e - diag_X
        Kt_YY_sums = K_YY.sum(dim=1) - diag_Y  # \tilde{K}_YY * e = K_YY * e - diag_Y
        K_XY_sums_0 = K_XY.sum(dim=0)  # K_{XY}^T * e

        Kt_XX_sum = Kt_XX_sums.sum()  # e^T * \tilde{K}_XX * e
        Kt_YY_sum = Kt_YY_sums.sum()  # e^T * \tilde{K}_YY * e
        K_XY_sum = K_XY_sums_0.sum()  # e^T * K_{XY} * e

        if biased:
            mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                    + (Kt_YY_sum + sum_diag_Y) / (m * m)
                    - 2.0 * K_XY_sum / (m * m))
        else:
            mmd2 = (Kt_XX_sum / (m * (m - 1))
                    + Kt_YY_sum / (m * (m - 1))
                    - 2.0 * K_XY_sum / (m * m))

        return mmd2

    def mmd(self, x, y):
        sigma_list = [sigma / self.base for sigma in self.scales]

        K_XX, K_XY, K_YY, _ = self._mix_rbf_kernel(x, y, sigma_list)
        mmd2_D = self._mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=self.biased)
        return mmd2_D

    def forward_da(self, a, b, c):
        a = self.hidden_features(a).view(a.size[0], -1)
        b = self.hidden_features(b).view(b.size[0], -1)
        c = self.hidden_features(c).view(c.size[0], -1)

        return (
                       self.mmd(a, b) +
                       self.mmd(b, c) +
                       self.mmd(c, a)
               ) / 3



