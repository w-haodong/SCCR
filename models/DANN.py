import torch
import torch.nn as nn
from torch.autograd import Function

# -----------------------------------------------------------------------------
# 1. 梯度反转的 autograd.Function (核心实现)
# -----------------------------------------------------------------------------
class GradientReversalFunction(Function):
    """
    梯度反转层的核心实现，继承自torch.autograd.Function。
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        """
        正向传播：记录lambda_值，并原样返回输入x。
        """
        # 将lambda_保存在ctx中，以便在反向传播时使用
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：将上游传来的梯度乘以-lambda_。
        """
        # grad_output是上一层传回的梯度
        # output是我们将要传给下一层的、方向反转后的梯度
        output = grad_output.neg() * ctx.lambda_
        
        # forward的输入有(x, lambda_)两个，所以backward的输出也需要对应两个梯度。
        # lambda_本身不需要梯度，所以返回None。
        return output, None

# -----------------------------------------------------------------------------
# 2. 梯度反转的 nn.Module (封装层)
# -----------------------------------------------------------------------------
class GradientReversalLayer(nn.Module):
    """
    【推荐的封装模式】将GradientReversalFunction封装为nn.Module。
    在forward方法中直接接收动态的lambda_参数。
    """
    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def forward(self, x, lambda_):
        # 直接调用Function的apply方法，并将lambda_作为参数传入
        return GradientReversalFunction.apply(x, lambda_)

# -----------------------------------------------------------------------------
# 3. 域分类器 (保持不变)
# -----------------------------------------------------------------------------
class DomainClassifier(nn.Module):
    """
    一个简单的域分类器。
    它接收展平的特征，并输出一个logits值（未经sigmoid）。
    """
    def __init__(self, input_features, hidden_features=256):
        super(DomainClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_features, 1)
        )

    def forward(self, x):
        return self.classifier(x)