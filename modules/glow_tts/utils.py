import numpy as np
import torch
import torch.nn.functional as F


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def shift_1d(x):
    x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
    return x


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def maximum_path(value, mask, max_neg_val=-np.inf):
    """ Numpy-friendly version. It's about 4 times faster than torch version.
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    """
    value = value * mask

    device = value.device
    dtype = value.dtype
    value = value.cpu().detach().numpy()
    mask = mask.cpu().detach().numpy().astype(np.bool)

    b, t_x, t_y = value.shape
    direction = np.zeros(value.shape, dtype=np.int64)
    v = np.zeros((b, t_x), dtype=np.float32)
    x_range = np.arange(t_x, dtype=np.float32).reshape(1, -1)
    for j in range(t_y):
        v0 = np.pad(v, [[0, 0], [1, 0]], mode="constant", constant_values=max_neg_val)[:, :-1]
        v1 = v
        max_mask = (v1 >= v0)
        v_max = np.where(max_mask, v1, v0)
        direction[:, :, j] = max_mask

        index_mask = (x_range <= j)
        v = np.where(index_mask, v_max + value[:, :, j], max_neg_val)
    direction = np.where(mask, direction, 1)

    path = np.zeros(value.shape, dtype=np.float32)
    index = mask[:, :, 0].sum(1).astype(np.int64) - 1
    index_range = np.arange(b)
    for j in reversed(range(t_y)):
        path[index_range, index, j] = 1
        index = index + direction[index_range, index, j] - 1
    path = path * mask.astype(np.float32)
    path = torch.from_numpy(path).to(device=device, dtype=dtype)
    return path


def generate_path(duration, mask):
    """
    duration: [b, t_x]
    mask: [b, t_x, t_y]
    """
    device = duration.device

    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path * mask
    return path


def squeeze(x, x_mask=None, n_sqz=2):
    b, c, t = x.size()

    t = (t // n_sqz) * n_sqz
    x = x[:, :, :t]
    x_sqz = x.view(b, c, t // n_sqz, n_sqz)
    x_sqz = x_sqz.permute(0, 3, 1, 2).contiguous().view(b, c * n_sqz, t // n_sqz)

    if x_mask is not None:
        x_mask = x_mask[:, :, n_sqz - 1::n_sqz]
    else:
        x_mask = torch.ones(b, 1, t // n_sqz).to(device=x.device, dtype=x.dtype)
    return x_sqz * x_mask, x_mask


def unsqueeze(x, x_mask=None, n_sqz=2):
    b, c, t = x.size()

    x_unsqz = x.view(b, n_sqz, c // n_sqz, t)
    x_unsqz = x_unsqz.permute(0, 2, 3, 1).contiguous().view(b, c // n_sqz, t * n_sqz)

    if x_mask is not None:
        x_mask = x_mask.unsqueeze(-1).repeat(1, 1, 1, n_sqz).view(b, 1, t * n_sqz)
    else:
        x_mask = torch.ones(b, 1, t * n_sqz).to(device=x.device, dtype=x.dtype)
    return x_unsqz * x_mask, x_mask
