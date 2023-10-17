import math
import torch
import random
import torchvision.transforms as transforms


class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, input):
        out = [transforms.functional.resize(input[:,i,:,:,:], self.output_size) for i in range(input.shape[1])]
        out = torch.stack(out).permute(1,0,2,3,4)
        return out


"""
Imported from pytorchvideo https://pytorchvideo.org/
"""



class ToTensor(object):

    def __call__(self, video):
        return torch.Tensor(video)


def _get_param_spatial_crop(scale, ratio, height, width, log_uniform_ratio=True, num_tries=10):
    """
    Given scale, ratio, height and width, return sampled coordinates of the videos.

    Args:
        scale (Tuple[float, float]): Scale range of Inception-style area based
            random resizing.
        ratio (Tuple[float, float]): Aspect ratio range of Inception-style
            area based random resizing.
        height (int): Height of the original image.
        width (int): Width of the original image.
        log_uniform_ratio (bool): Whether to use a log-uniform distribution to
            sample the aspect ratio. Default is True.
        num_tries (int): The number of times to attempt a randomly resized crop.
            Falls back to a central crop after all attempts are exhausted.
            Default is 10.

    Returns:
        Tuple containing i, j, h, w. (i, j) are the coordinates of the top left
        corner of the crop. (h, w) are the height and width of the crop.
    """
    assert num_tries >= 1, "num_tries must be at least 1"

    if scale[0] > scale[1]:
        scale = (scale[1], scale[0])
    if ratio[0] > ratio[1]:
        ratio = (ratio[1], ratio[0])

    for _ in range(num_tries):
        area = height * width
        target_area = area * (scale[0] + torch.rand(1).item() * (scale[1] - scale[0]))
        if log_uniform_ratio:
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(
                log_ratio[0] + torch.rand(1).item() * (log_ratio[1] - log_ratio[0])
            )
        else:
            aspect_ratio = ratio[0] + torch.rand(1).item() * (ratio[1] - ratio[0])

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if 0 < w <= width and 0 < h <= height:
            i = torch.randint(0, height - h + 1, (1,)).item()
            j = torch.randint(0, width - w + 1, (1,)).item()
            return i, j, h, w

    # Fallback to central crop.
    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
        w = width
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height
        w = int(round(h * max(ratio)))
    else:  # whole image
        w = width
        h = height
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w

def random_resized_crop(frames, target_height, target_width, scale, aspect_ratio, shift=False, log_uniform_ratio=True, interpolation="bilinear", num_tries=10):
    """
    Crop the given images to random size and aspect ratio. A crop of random
    size relative to the original size and a random aspect ratio is made. This
    crop is finally resized to given size. This is popularly used to train the
    Inception networks.

    Args:
        frames (torch.Tensor): Video tensor to be resized with shape (T, H, W, C).
        target_height (int): Desired height after cropping.
        target_width (int): Desired width after cropping.
        scale (Tuple[float, float]): Scale range of Inception-style area based
            random resizing. Should be between 0.0 and 1.0.
        aspect_ratio (Tuple[float, float]): Aspect ratio range of Inception-style
            area based random resizing. Should be between 0.0 and +infinity.
        shift (bool): Bool that determines whether or not to sample two different
            boxes (for cropping) for the first and last frame. If True, it then
            linearly interpolates the two boxes for other frames. If False, the
            same box is cropped for every frame. Default is False.
        log_uniform_ratio (bool): Whether to use a log-uniform distribution to
            sample the aspect ratio. Default is True.
        interpolation (str): Algorithm used for upsampling. Currently supports
            'nearest', 'bilinear', 'bicubic', 'area'. Default is 'bilinear'.
        num_tries (int): The number of times to attempt a randomly resized crop.
            Falls back to a central crop after all attempts are exhausted.
            Default is 10.

    Returns:
        cropped (tensor): A cropped video tensor of shape (T, target_height, target_width, C).
    """
    frames = frames.permute(3,0,1,2)
    assert (
        scale[0] > 0 and scale[1] > 0
    ), "min and max of scale range must be greater than 0"
    assert (
        aspect_ratio[0] > 0 and aspect_ratio[1] > 0
    ), "min and max of aspect_ratio range must be greater than 0"

    channels = frames.shape[0]
    t = frames.shape[1]
    height = frames.shape[2]
    width = frames.shape[3]

    i, j, h, w = _get_param_spatial_crop(
        scale, aspect_ratio, height, width, log_uniform_ratio, num_tries
    )

    if not shift:
        cropped = frames[:, :, i : i + h, j : j + w]
        return torch.nn.functional.interpolate(
            cropped,
            size=(target_height, target_width),
            mode=interpolation,
            align_corners=True,
        ).permute(1,2,3,0)

    i_, j_, h_, w_ = _get_param_spatial_crop(
        scale, aspect_ratio, height, width, log_uniform_ratio, num_tries
    )
    i_s = [int(i) for i in torch.linspace(i, i_, steps=t).tolist()]
    j_s = [int(i) for i in torch.linspace(j, j_, steps=t).tolist()]
    h_s = [int(i) for i in torch.linspace(h, h_, steps=t).tolist()]
    w_s = [int(i) for i in torch.linspace(w, w_, steps=t).tolist()]
    cropped = torch.zeros((channels, t, target_height, target_width))
    for ind in range(t):
        cropped[:, ind : ind + 1, :, :] = torch.nn.functional.interpolate(
            frames[
                :,
                ind : ind + 1,
                i_s[ind] : i_s[ind] + h_s[ind],
                j_s[ind] : j_s[ind] + w_s[ind],
            ],
            size=(target_height, target_width),
            mode=interpolation,
            align_corners=True,
        )
    return cropped.permute(1,2,3,0)


class RandomResizedCrop(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.random_resized_crop``.
    """

    def __init__(
        self,
        target_height,
        target_width,
        scale,
        aspect_ratio,
        shift=False,
        log_uniform_ratio=True,
        interpolation="bilinear",
        num_tries=10,
    ):

        super().__init__()
        self._target_height = target_height
        self._target_width = target_width
        self._scale = scale
        self._aspect_ratio = aspect_ratio
        self._shift = shift
        self._log_uniform_ratio = log_uniform_ratio
        self._interpolation = interpolation
        self._num_tries = num_tries

    def __call__(self, x):
        """
        Args:
            x (torch.Tensor): Input video tensor with shape (T, H, W, C).
        """
        return random_resized_crop(
            x,
            self._target_height,
            self._target_width,
            self._scale,
            self._aspect_ratio,
            self._shift,
            self._log_uniform_ratio,
            self._interpolation,
            self._num_tries,
        )


class RandomHorizontalFlip(object):
    """ Horizontal flip the given video tensor (L x H x W x C) randomly with a given probability.
    Args:
        p (float): probability of the video being flipped. Default value is 0.5.

    Reference:
        https://github.com/YuxinZhaozyx/pytorch-VideoDataset/blob/master/transforms.py
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video to flipped.

        Returns:
            torch.Tensor: Randomly flipped video.
        """

        if random.random() < self.p:
            # horizontal flip the video
            video = video.flip([2])

        return video


def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tesnor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True

def normalize(clip, mean, std, inplace=False):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (T, H, W, C)
        mean (tuple): pixel RGB mean. Size is (3)
        std (tuple): pixel standard deviation. Size is (3)
    Returns:
        normalized clip (torch.tensor): Size is (T, H, W, C)
    """
    assert _is_tensor_video_clip(clip), "clip should be a 4D torch.tensor"
    if not inplace:
        clip = clip.clone()
    mean = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=clip.dtype, device=clip.device)
    clip.sub_(mean[None, None, None, :]).div_(std[None, None, None, :])
    return clip
class NormalizeVideo(object):
    """
    Normalize the video clip by mean subtraction
    and division by standard deviation
    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): video clip to be
                                normalized. Size is (T, H, W, C)
        """
        return normalize(clip, self.mean, self.std, self.inplace)