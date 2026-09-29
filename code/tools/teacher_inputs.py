"""Teacher-side inputs matched to the information available to DVS."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def grayscale(image: torch.Tensor) -> torch.Tensor:
    if image.shape[1] == 1:
        return image
    weights = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (image[:, :3] * weights).sum(dim=1, keepdim=True)


def sobel(gray: torch.Tensor) -> torch.Tensor:
    kernel_x = gray.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    gx = F.conv2d(gray, kernel_x, padding=1)
    gy = F.conv2d(gray, kernel_y, padding=1)
    magnitude = torch.sqrt(gx.square() + gy.square() + 1.0e-6)
    scale = torch.quantile(magnitude.flatten(1), 0.99, dim=1).view(-1, 1, 1, 1).clamp_min(1.0e-6)
    return (magnitude / scale).clamp(0.0, 1.0)


def canny_like(gray: torch.Tensor) -> torch.Tensor:
    """Dependency-free Canny-style edge carrier for teacher ablations.

    It uses Sobel magnitude, non-maximum-like local suppression, and adaptive
    hysteresis thresholds. The exact sensor event stream remains unchanged.
    """
    magnitude = sobel(gray)
    local_max = F.max_pool2d(magnitude, kernel_size=3, stride=1, padding=1)
    thin = magnitude * (magnitude >= 0.9 * local_max).to(magnitude.dtype)
    high = torch.quantile(thin.flatten(1), 0.90, dim=1).view(-1, 1, 1, 1)
    low = 0.45 * high
    strong = thin >= high
    weak = thin >= low
    connected = F.max_pool2d(strong.float(), kernel_size=3, stride=1, padding=1) > 0
    return (weak & connected).to(gray.dtype)


def make_teacher_input(image: torch.Tensor, mode: str) -> torch.Tensor:
    """Return a 3-channel teacher image without inventing color information."""
    if mode == "rgb":
        return image
    gray = grayscale(image)
    if mode == "gray":
        return gray.expand(-1, 3, -1, -1)
    if mode == "sobel":
        edge = sobel(gray)
        return edge.expand(-1, 3, -1, -1)
    if mode == "canny":
        edge = canny_like(gray)
        return edge.expand(-1, 3, -1, -1)
    if mode == "gray_sobel":
        edge = sobel(gray)
        return torch.cat([gray, edge, gray * (1.0 - edge)], dim=1)
    raise ValueError(f"unsupported teacher input mode: {mode}")
