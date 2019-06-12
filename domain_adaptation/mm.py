import torch


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
