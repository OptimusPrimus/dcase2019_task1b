import torch


class MutualInformationDA(torch.nn.Module):

    def __init__(self, model, nr_clusters=10):
        super(MutualInformationDA, self).__init__()

        self.model = model
        self.nr_clusters = nr_clusters
        self.hidden_features = torch.nn.Sequential(
                *list(model.children())[:-1]
            )

        in_channels = list(list(model.children())[-1].children())[0].in_channels

        self.cluster = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels,
                nr_clusters,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            torch.nn.AdaptiveAvgPool2d((1,1)),
            torch.nn.Softmax(dim=1)
        )

        self.cluster.apply(self.initialize_weights)

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity="relu")
        elif isinstance(module, torch.nn.BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()
        elif isinstance(module, torch.nn.Linear):
            module.bias.data.zero_()

    def forward(self, x):
        return self.model(x)

    def forward_da(self, a, b, c):

        a = self.cluster(self.hidden_features(a))
        b = self.cluster(self.hidden_features(b))
        c = self.cluster(self.hidden_features(c))

        i_a = a.view(a.shape[0], -1)
        i_b = b.view(b.shape[0], -1)
        i_c = c.view(c.shape[0], -1)

        a = self.IIC(i_a, i_b, C=self.nr_clusters)
        b = self.IIC(i_b, i_c, C=self.nr_clusters)
        c = self.IIC(i_c, i_a, C=self.nr_clusters)

        return (a + b + c) / (3* i_a.shape[0])

    @staticmethod
    def IIC(z, zt, C=10, eps=1e-10):
        P = (z.unsqueeze(2) * zt.unsqueeze(1)).sum(dim=0)
        # ????
        P = ((P + P.t()) / 2) / P.sum()

        P[(P < eps).data] = eps
        Pi = P.sum(dim=1).view(C, 1).expand(C, C)
        Pj = P.sum(dim=0).view(1, C).expand(C, C)
        return (P * (torch.log(Pi) + torch.log(Pj) - torch.log(P))).sum()
