import torch
from torch.nn import Module


class Quantizer(Module):
    def __init__(self, bins, min_v, max_v):
        super(Quantizer, self).__init__()
        self.bins = bins
        self.min_v = min_v
        self.max_v = max_v

        d = (self.max_v - self.min_v) / (self.bins - 1)  # distance between bin centers
        k = 1 / d                                        # slope of quantization lines

        # Conv1d can be used to calculate lines: ax + b, where a: weights, b: biases
        self.q = torch.nn.Conv1d(1, 2 * bins, kernel_size=(1,), bias=True)

        # Descending lines: y=1 at bin center x=c, crossing zero (y=0) at x=c+d with a slope of -1/d,
        #   where `d` is the distance to the next bin center.
        # Line equation is y = a * x + b, where a = -k, x = c = i * d + min_v, and `i` is the bin index {0 ... bins-1}.
        # Solving for the bias `b`:
        #   1 = -k * (d * i + min_v) + b  =>  b = 1 + k * (d * i + min_v) = 1 + k * min_v + k * d * i
        self.q.weight.data[:bins] = -k
        self.q.bias.data[:bins] = 1 + k * min_v + k * d * torch.arange(bins-1, -0.01, -1)

        # Ascending lines: crossing zero (y=0) at x=c-d, rising to y=1 at bin center x=c.
        # Solving `y = a * x + b` for bias `b`:
        #   1 = k * (d * i + min_v) + b   =>  b = 1 - k * (d * i + min_v) = 1 - k * min_v - k * d * i
        self.q.weight.data[bins:] = k
        self.q.bias.data[bins:] = 1 - k * min_v - k * d * torch.arange(bins-1, -0.01, -1)

        # First bin (centered at max_v) collects all values > max_v   => instead of a descending line, a horizontal one
        self.q.weight.data[0] = 0
        self.q.bias.data[0] = 1

        # Last bin (centered at min_v) collects all values < min_v    => instead of ascending line, a horizontal one
        self.q.weight.data[-1] = 0
        self.q.bias.data[-1] = 1

    def forward(self, x, insert_dim=1):
        binned_x = self.q(x[:, None, ...])
        binned_x = torch.min(binned_x[:, :self.bins, ...], binned_x[:, self.bins:, ...]).clamp(min=0)
        ord = list(range(0, len(binned_x.shape)))
        ord.remove(1)
        ord.insert(insert_dim, 1)
        return binned_x.permute(ord)
