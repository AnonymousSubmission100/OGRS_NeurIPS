import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def weights_init_normal(m):
    """Initializes the weight and bias of the model.

    Args:
        m: A torch model to initialize.

    Returns:
        None.
    """
    
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.2)
        torch.nn.init.constant_(m.bias.data, 0)

# Second place need to change, need to define a new BNN model
class LogisticRegression(nn.Module):
    """Logistic Regression (classifier).

    Attributes:
        model: A model consisting of torch components.
    """

    def __init__(self, n_in, n_out):
        """Initializes classifier with torch components."""

        super(LogisticRegression, self).__init__()


        def block(in_feat, out_feat, normalize=True):
            """Defines a block with torch components.

                Args:
                    in_feat: An integer value for the size of the input feature.
                    out_feat: An integer value for the size of the output feature.
                    normalize: A boolean indicating whether normalization is needed.

                Returns:
                    The stacked layer.
            """

            layers = [nn.Linear(in_feat, out_feat)]

            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            nn.Linear(n_in,n_out)
        )

    def forward(self, input_data):
        """Defines a forward operation of the model.

        Args:
            input_data: The input data.

        Returns:
            The predicted label (y_hat) for the given input data.
        """

        output = self.model(input_data)
        return output
#
# class LogisticRegression(torch.nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(LogisticRegression, self).__init__()
#         self.linear = torch.nn.Linear(input_dim, output_dim)
#
#     def forward(self, x):
#         outputs = torch.sigmoid(self.linear(x))
#         return outputs
class BasicNet(torch.nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.LazyLinear(num_outputs))
    def forward(self, X):
        return self.net(X)

def test_model(model_, X, y):
    """Tests the performance of a model.

    Args:
        model_: A model to test.
        X: Input features of test data.
        y: True label (1-D) of test data.
        s1: Sensitive attribute (1-D) of test data.

    Returns:
        The test accuracy and the fairness metrics of the model.
    """
    
    model_.eval()
    
    y_hat = model_(X).squeeze()
    prediction = (y_hat > 0.0).int().squeeze()
    y = (y > 0.0).int()
    test_acc = torch.sum(prediction == y.int()).float() / len(y)

    return test_acc.item()
    # return {'Acc': test_acc.item(), 'DP_diff': DP, 'EO_Y0_diff': EO_Y_0, 'EO_Y1_diff': EO_Y_1, 'EqOdds_diff': max(EO_Y_0, EO_Y_1)}



