# utils/time2vec.py
import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    def __init__(self, dim: int = 32):
        """
        Time2Vec 时间嵌入模块
        :param dim: 输出维度 (包含一个线性分量 + 多个正余弦分量)
        """
        super(Time2Vec, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(1, 1)                     # 线性部分 w0 * t + b0
        self.freq_weights = nn.Parameter(torch.randn(dim - 1))  # 周期部分 w_i
        self.freq_biases = nn.Parameter(torch.randn(dim - 1))   # 周期部分 b_i

    def forward(self, time_input):
        """
        :param time_input: pandas.Timestamp or numpy.datetime64
        :return: torch.Tensor of shape (dim)
        """
        if not torch.is_tensor(time_input):
            # 将 datetime 转为 UNIX timestamp (float 秒数)
            time_input = torch.tensor([[time_input.timestamp()]], dtype=torch.float32)
        else:
            time_input = time_input.view(1, 1)

        # 线性项
        linear_out = self.linear(time_input)  # (1, 1)

        # 正余弦周期项
        freq_out = torch.sin(time_input * self.freq_weights + self.freq_biases)  # (1, dim - 1)

        return torch.cat([linear_out, freq_out], dim=1).squeeze()  # (dim,)