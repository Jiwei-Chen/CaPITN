import copy
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn


class ModelTimeRecorder:
    """Record model training and inference wall-clock time."""

    def __init__(self, device=None):
        self.device = torch.device(device) if device is not None else None
        self.training_time = 0.0
        self.inference_time = 0.0

    def _synchronize(self):
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def record_training(self):
        self._synchronize()
        start_time = time.perf_counter()
        try:
            yield
        finally:
            self._synchronize()
            self.training_time += time.perf_counter() - start_time

    @contextmanager
    def record_inference(self):
        self._synchronize()
        start_time = time.perf_counter()
        try:
            yield
        finally:
            self._synchronize()
            self.inference_time += time.perf_counter() - start_time


class RelativeL2ConvergenceTracker:
    """Track relative L2 error against cumulative training time."""

    def __init__(self, reference, threshold):
        try:
            self.reference = np.asarray(reference, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("reference must contain finite numeric values") from error

        if self.reference.size == 0:
            raise ValueError("reference must not be empty")
        if not np.all(np.isfinite(self.reference)):
            raise ValueError("reference must contain finite values")

        self.reference_norm = np.linalg.norm(self.reference)
        if self.reference_norm == 0:
            raise ValueError("reference must have a nonzero L2 norm")

        try:
            threshold = float(threshold)
        except (TypeError, ValueError) as error:
            raise ValueError("threshold must be a finite nonnegative value") from error

        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be a finite nonnegative value")

        self.threshold = threshold
        self.history = []
        self.first_threshold_time = None

    def record(self, training_time, prediction):
        """Record one [training_time, relative_l2_error] sample."""
        try:
            training_time = float(training_time)
        except (TypeError, ValueError) as error:
            raise ValueError("training_time must be a finite nonnegative value") from error

        if not np.isfinite(training_time) or training_time < 0:
            raise ValueError("training_time must be a finite nonnegative value")
        if self.history and training_time < self.history[-1][0]:
            raise ValueError("training_time must be nondecreasing")

        try:
            prediction = np.asarray(prediction, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("prediction must contain finite numeric values") from error

        if prediction.shape != self.reference.shape:
            raise ValueError("prediction and reference must have the same shape")
        if not np.all(np.isfinite(prediction)):
            raise ValueError("prediction must contain finite values")

        relative_l2_error = np.linalg.norm(prediction - self.reference) / self.reference_norm
        self.history.append([training_time, relative_l2_error])

        if (
            self.first_threshold_time is None
            and relative_l2_error <= self.threshold
        ):
            self.first_threshold_time = training_time

        return relative_l2_error

    def as_array(self):
        """Return samples as [training_time, relative_l2_error]."""
        return np.asarray(self.history, dtype=np.float64).reshape(-1, 2)


def get_data(x_range, y_range, x_num, y_num):
    x = np.linspace(x_range[0], x_range[1], x_num)
    t = np.linspace(y_range[0], y_range[1], y_num)

    x_mesh, t_mesh = np.meshgrid(x,t)
    data = np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1)
    
    b_left = data[0,:,:] 
    b_right = data[-1,:,:]
    b_upper = data[:,-1,:]
    b_lower = data[:,0,:]
    res = data.reshape(-1,2)

    return res, b_left, b_right, b_upper, b_lower


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def make_time_sequence(src, num_step=5, step=1e-4):
    dim = num_step
    src = np.repeat(np.expand_dims(src, axis=1), dim, axis=1).astype(np.float64)  # (N, L, 2)
    for i in range(num_step):
        src[:,i,-1] += step*i
    return src


def make_pseudo_spacetime_patch(src, num_space=3, space_step=1e-2, num_step=3, step=1e-4):
    """
    Construct a fixed pseudo-spatiotemporal candidate patch.

    src: [N, D+1], with spatial coordinates followed by time.
    num_space: odd number of offsets along each spatial dimension.
    space_step: distance between neighboring spatial offsets.
    num_step: number of temporal offsets.
    step: temporal offset interval.

    Returns: [N, num_step * num_space**D, D+1]
    """
    src = np.asarray(src, dtype=np.float64)

    if src.ndim != 2 or src.shape[1] < 2:
        raise ValueError("src must have shape [N, D_space+1]")

    if num_space < 1 or num_space % 2 == 0:
        raise ValueError("num_space must be a positive odd integer")

    if num_step < 1:
        raise ValueError("num_step must be a positive integer")

    D = src.shape[1] - 1
    spatial_axis_offsets = (np.arange(num_space) - num_space // 2) * space_step
    spatial_offsets = np.stack(
        np.meshgrid(*([spatial_axis_offsets] * D), indexing="ij"),
        axis=-1,
    ).reshape(-1, D)
    time_offsets = -np.arange(num_step - 1, -1, -1) * step

    N = src.shape[0]
    Ns = spatial_offsets.shape[0]
    patch = np.broadcast_to(src[:, None, None, :], (N, num_step, Ns, D + 1)).copy()
    patch[..., :D] += spatial_offsets[None, None, :, :]
    patch[..., -1] += time_offsets[None, :, None]

    return patch.reshape(N, num_step * Ns, D + 1)


def make_pseudo_spacetime_cross_patch(
    src,
    num_space=3,
    space_step=1e-2,
    num_step=3,
    step=1e-4,

):
    """Construct a pseudo-spatiotemporal cross patch in any spatial dimension.

    ``num_space`` is the odd number of points along each spatial axis.
    ``space_step`` may be a scalar or a sequence with one value per spatial
    dimension. For 2D and ``num_space=5``, each temporal layer is ordered as:
    top2, left2, top1, left1, center, right1, bottom1, right2, bottom2.

    src: [N, D+1], with spatial coordinates followed by time.
    Returns: [N, num_step * (1 + D*(num_space-1)), D+1].
    """
    src = np.asarray(src, dtype=np.float64)

    if src.ndim != 2 or src.shape[1] < 2:
        raise ValueError("src must have shape [N, D_space+1]")

    if num_step < 1:
        raise ValueError("num_step must be a positive integer")

    if num_space < 1 or num_space % 2 == 0:
        raise ValueError("num_space must be a positive odd integer")

    D = src.shape[1] - 1
    spacing = np.asarray(space_step, dtype=np.float64)
    if spacing.ndim == 0:
        spacing = np.full(D, float(spacing))
    elif spacing.shape != (D,):
        raise ValueError(
            f"space_step must be a scalar or a length-{D} sequence"
        )

    if np.any(spacing <= 0):
        raise ValueError("spatial step sizes must be positive")

    positive_directions = np.diag(spacing)
    if D >= 2:
        positive_directions[1] *= -1
    negative_directions = -positive_directions[::-1]

    radius = num_space // 2
    spatial_offsets = [
        level * direction
        for level in range(radius, 0, -1)
        for direction in negative_directions
    ]
    spatial_offsets.append(np.zeros(D))
    spatial_offsets.extend(
        level * direction
        for level in range(1, radius + 1)
        for direction in positive_directions
    )
    spatial_offsets = np.asarray(spatial_offsets, dtype=np.float64)

    time_offsets = -np.arange(num_step - 1, -1, -1) * step
    N = src.shape[0]
    num_spatial_tokens = spatial_offsets.shape[0]
    patch = np.broadcast_to(
        src[:, None, None, :], (N, num_step, num_spatial_tokens, D + 1)
    ).copy()
    patch[..., :D] += spatial_offsets[None, None, :, :]
    patch[..., -1] += time_offsets[None, :, None]

    return patch.reshape(N, num_step * num_spatial_tokens, D + 1)


def get_pseudo_spacetime_cross_center_index(space_dim, num_space=3, num_step=3):
    """Return the latest-time center index of a flattened cross patch."""
    if not isinstance(space_dim, (int, np.integer)) or space_dim < 1:
        raise ValueError("space_dim must be a positive integer")

    if num_space < 1 or num_space % 2 == 0:
        raise ValueError("num_space must be a positive odd integer")

    if num_step < 1:
        raise ValueError("num_step must be a positive integer")

    num_spatial_tokens = 1 + int(space_dim) * (num_space - 1)
    return (num_step - 1) * num_spatial_tokens + num_spatial_tokens // 2


def make_pseudo_spacetime_cross_offsets(
    space_dim,
    num_space=3,
    space_step=1e-2,
    num_step=3,
    step=1e-4,
):
    """Return fixed offsets for a cross patch anchored at its latest center."""
    if not isinstance(space_dim, (int, np.integer)) or space_dim < 1:
        raise ValueError("space_dim must be a positive integer")
    center = np.zeros((1, int(space_dim) + 1), dtype=np.float64)
    return make_pseudo_spacetime_cross_patch(
        center,
        num_space=num_space,
        space_step=space_step,
        num_step=num_step,
        step=step,
    )[0]


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_data_3d(x_range, y_range, t_range, x_num, y_num, t_num):
    step_x = (x_range[1] - x_range[0]) / float(x_num-1)
    step_y = (y_range[1] - y_range[0]) / float(y_num-1)
    step_t = (t_range[1] - t_range[0]) / float(t_num-1)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[0]:x_range[1]+step_x:step_x,y_range[0]:y_range[1]+step_y:step_y,t_range[0]:t_range[1]+step_t:step_t]

    data = np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1)
    res = data.reshape(-1,3)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[0]:x_range[0]+step_x:step_x,y_range[0]:y_range[1]+step_y:step_y,t_range[0]:t_range[1]+step_t:step_t]
    b_left = np.squeeze(np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1))[1:-1].reshape(-1,3)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[1]:x_range[1]+step_x:step_x,y_range[0]:y_range[1]+step_y:step_y,t_range[0]:t_range[1]+step_t:step_t]
    b_right = np.squeeze(np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1))[1:-1].reshape(-1,3)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[0]:x_range[1]+step_x:step_x,y_range[0]:y_range[0]+step_y:step_y,t_range[0]:t_range[1]+step_t:step_t]
    b_lower = np.squeeze(np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1))[1:-1].reshape(-1,3)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[0]:x_range[1]+step_x:step_x,y_range[1]:y_range[1]+step_y:step_y,t_range[0]:t_range[1]+step_t:step_t]
    b_upper = np.squeeze(np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1))[1:-1].reshape(-1,3)

    return res, b_left, b_right, b_upper, b_lower

def data_processing(nodes, time, U, P):
    x = np.tile(nodes[:,0:1], (time.shape[0],1))
    y = np.tile(nodes[:,1:2], (time.shape[0],1))
    t = np.tile(time, (1,nodes.shape[0])).flatten()[:,None]

    u = U[:,:,0].reshape(-1, 1)
    v = U[:,:,1].reshape(-1, 1)
    p = P[:,:,0].reshape(-1, 1)

    return x, y, t, u, v, p

