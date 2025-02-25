# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Implement many useful :class:`Augmentation`.
"""
import numpy as np
import sys
from fvcore.transforms.transform import (
    BlendTransform,
    CropTransform,
    HFlipTransform,
    NoOpTransform,
    Transform,
    VFlipTransform,
)
from PIL import Image

from .augmentation import Augmentation
from .transform import ExtentTransform, ResizeTransform, RotationTransform

__all__ = [
    "RandomApply",
    "RandomBrightness",
    "RandomContrast",
    "RandomCrop",
    "RandomExtent",
    "RandomFlip",
    "RandomSaturation",
    "RandomLighting",
    "RandomRotation",
    "Resize",
    "ResizeShortestEdge",
]


class RandomApply(Augmentation):
    """
    Randomly apply the wrapper transformation with a given probability.
    """

    def __init__(self, transform, prob=0.5):
        """
        Args:
            transform (Transform, Augmentation): the transform to be wrapped
                by the `RandomApply`. The `transform` can either be a
                `Transform` or `Augmentation` instance.
            prob (float): probability between 0.0 and 1.0 that
                the wrapper transformation is applied
        """
        super().__init__()
        assert isinstance(transform, (Transform, Augmentation)), (
            f"The given transform must either be a Transform or Augmentation instance. "
            f"Not {type(transform)}"
        )
        assert 0.0 <= prob <= 1.0, f"Probablity must be between 0.0 and 1.0 (given: {prob})"
        self.prob = prob
        self.transform = transform
        if isinstance(transform, Augmentation):
            self.input_args = transform.input_args
        self._init_random_params(('do',))

    def randomize_parameters(self):
        self.set_random_param('do', self._rand_range() < self.prob)

    def get_transform(self, img):
        do = self.get_random_param('do')
        if do is None:
            do = self._rand_range() < self.prob
        
        if do:
            if isinstance(self.transform, Augmentation):
                return self.transform.get_transform(img)
            else:
                return self.transform
        else:
            return NoOpTransform()


class RandomFlip(Augmentation):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(self, prob=0.5, *, horizontal=True, vertical=False):
        """
        Args:
            prob (float): probability of flip.
            horizontal (boolean): whether to apply horizontal flipping
            vertical (boolean): whether to apply vertical flipping
        """
        super().__init__()

        if horizontal and vertical:
            raise ValueError("Cannot do both horiz and vert. Please use two Flip instead.")
        if not horizontal and not vertical:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._init(locals())
        self._init_random_params(('do',))

    def randomize_parameters(self):
        self.set_random_param('do', self._rand_range() < self.prob)

    def get_transform(self, img):
        do = self.get_random_param('do')
        if do is None:
            do = self._rand_range() < self.prob

        h, w = img.shape[:2]

        if do:
            if self.horizontal:
                return HFlipTransform(w)
            elif self.vertical:
                return VFlipTransform(h)
        else:
            return NoOpTransform()


class Resize(Augmentation):
    """ Resize image to a fixed target size"""

    def __init__(self, shape, interp=Image.BILINEAR):
        """
        Args:
            shape: (h, w) tuple or a int
            interp: PIL interpolation method
        """
        if isinstance(shape, int):
            shape = (shape, shape)
        shape = tuple(shape)
        self._init(locals())
        self._init_random_params(())

    def randomize_parameters(self):
        pass

    def get_transform(self, img):
        return ResizeTransform(
            img.shape[0], img.shape[1], self.shape[0], self.shape[1], self.interp
        )


class ResizeShortestEdge(Augmentation):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
        self, short_edge_length, max_size=sys.maxsize, sample_style="range", interp=Image.BILINEAR
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        self._init(locals())
        self._init_random_params(('size',))

    def randomize_parameters(self):
        self.set_random_param('size', self._randomize_size())

    def get_transform(self, img):
        size = self.get_random_param('size')
        if size is None:
            size = self._randomize_size()

        h, w = img.shape[:2]

        if size == 0:
            return NoOpTransform()

        scale = size * 1.0 / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return ResizeTransform(h, w, newh, neww, self.interp)
    
    def _randomize_size(self):
        if self.is_range:
            size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        else:
            size = np.random.choice(self.short_edge_length)
        return size


class RandomRotation(Augmentation):
    """
    This method returns a copy of this image, rotated the given
    number of degrees counter clockwise around the given center.
    """

    def __init__(self, angle, expand=True, center=None, sample_style="range", interp=None):
        """
        Args:
            angle (list[float]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the angle (in degrees).
                If ``sample_style=="choice"``, a list of angles to sample from
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (list[[float, float]]):  If ``sample_style=="range"``,
                a [[minx, miny], [maxx, maxy]] relative interval from which to sample the center,
                [0, 0] being the top left of the image and [1, 1] the bottom right.
                If ``sample_style=="choice"``, a list of centers to sample from
                Default: None, which means that the center of rotation is the center of the image
                center has no effect if expand=True because it only affects shifting
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style
        self.is_range = sample_style == "range"
        if isinstance(angle, (float, int)):
            angle = (angle, angle)
        if center is not None and isinstance(center[0], (float, int)):
            center = (center, center)
        self._init(locals())
        self._init_random_params(('angle', 'center'))

    def randomize_parameters(self):
        self.set_random_param('angle', self._randomize_angle())
        self.set_random_param('center', self._randomize_center())

    def get_transform(self, img):
        angle = self.get_random_param('angle')
        center = self.get_random_param('center')
        if angle is None:
            angle = self._randomize_angle()
        if center is None:
            center = self._randomize_center()

        h, w = img.shape[:2]

        if center is not None:
            center = (w * center[0], h * center[1])  # Convert to absolute coordinates
        if angle % 360 == 0:
            return NoOpTransform()
        return RotationTransform(h, w, angle, expand=self.expand, center=center, interp=self.interp)
    
    def _randomize_angle(self):
        if self.is_range:
            angle = np.random.uniform(self.angle[0], self.angle[1])
        else:
            angle = np.random.choice(self.angle)
        return angle
    
    def _randomize_center(self):
        center = None
        if self.center is not None:
            if self.is_range:
                center = (
                    np.random.uniform(self.center[0][0], self.center[1][0]),
                    np.random.uniform(self.center[0][1], self.center[1][1]),
                )
            else:
                center = np.random.choice(self.center)
        return center


class RandomCrop(Augmentation):
    """
    Randomly crop a subimage out of an image.
    """

    def __init__(self, crop_type: str, crop_size):
        """
        Args:
            crop_type (str): one of "relative_range", "relative", "absolute", "absolute_range".
                See `config/defaults.py` for explanation.
            crop_size (tuple[float]): the relative ratio or absolute pixels of
                height and width
        """
        super().__init__()
        assert crop_type in ["relative_range", "relative", "absolute", "absolute_range"]
        self._init(locals())
        self._init_random_params(('ch', 'cw', 'h0r', 'w0r'))

    def randomize_parameters(self):
        ch, cw = self._randomize_crop_size()
        self.set_random_param('ch', ch)
        self.set_random_param('cw', cw)

        h0r, w0r = self._randomize_crop_position()
        self.set_random_param('h0r', h0r)
        self.set_random_param('w0r', w0r)

    def get_transform(self, img):
        ch = self.get_random_param('ch')
        cw = self.get_random_param('cw')
        h0r = self.get_random_param('h0r')
        w0r = self.get_random_param('w0r')
        if ch is None or cw is None:
            ch, cw = self._randomize_crop_size()
        if h0r is None or w0r is None:
            h0r, w0r = self._randomize_crop_position()

        h, w = img.shape[:2]

        croph, cropw = self._get_crop_size((h, w), (ch, cw))
        assert h >= croph and w >= cropw, "Shape computation in {} has bugs.".format(self)
        h0 = int((h - croph) * h0r + 0.5)
        w0 = int((w - cropw) * h0r + 0.5)
        return CropTransform(w0, h0, cropw, croph)
    
    def _randomize_crop_size(self):
        if self.crop_type == "relative":
            return self.crop_size
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
            return ch, cw
        elif self.crop_type == "absolute":
            return crop_size
        elif self.crop_type == "absolute_range":
            assert self.crop_size[0] <= self.crop_size[1]
            ch = np.random.randint(self.crop_size[0], self.crop_size[1] + 1)
            cw = np.random.randint(self.crop_size[0], self.crop_size[1] + 1)
            return ch, cw
        else:
            NotImplementedError("Unknown crop type {}".format(self.crop_type))

    def _randomize_crop_position(self):
        h0r = np.random.rand()
        w0r = np.random.rand()
        return h0r, w0r

    def _get_crop_size(self, image_size, crop_size):
        """
        Args:
            image_size (tuple): height, width

        Returns:
            crop_size (tuple): height, width in absolute pixels
        """
        h, w = image_size
        ch, cw = crop_size
        if self.crop_type in ("relative", "relative_range"):
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type in ("absolute", "absolute_range"):
            return (min(ch, h), min(cw, w))
        else:
            NotImplementedError("Unknown crop type {}".format(self.crop_type))


class RandomExtent(Augmentation):
    """
    Outputs an image by cropping a random "subrect" of the source image.

    The subrect can be parameterized to include pixels outside the source image,
    in which case they will be set to zeros (i.e. black). The size of the output
    image will vary with the size of the random subrect.
    """

    def __init__(self, scale_range, shift_range):
        """
        Args:
            output_size (h, w): Dimensions of output image
            scale_range (l, h): Range of input-to-output size scaling factor
            shift_range (x, y): Range of shifts of the cropped subrect. The rect
                is shifted by [w / 2 * Uniform(-x, x), h / 2 * Uniform(-y, y)],
                where (w, h) is the (width, height) of the input image. Set each
                component to zero to crop at the image's center.
        """
        super().__init__()
        self._init(locals())
        self._init_random_params(('scale', 'sh', 'sw'))

    def randomize_parameters(self):
        scale = self._randomize_scale()
        self.set_random_param('scale', scale)

        sh, sw = self._randomize_shift()
        self.set_random_param('sh', sh)
        self.set_random_param('sw', sw)

    def get_transform(self, img):
        scale = self.get_random_param('scale')
        sh = self.get_random_param('sh')
        sw = self.get_random_param('sw')
        if scale is None:
            scale = self._randomize_scale()
        if sh is None or sw is None:
            sh, sw = self._randomize_shift()

        img_h, img_w = img.shape[:2]

        # Initialize src_rect to fit the input image.
        src_rect = np.array([-0.5 * img_w, -0.5 * img_h, 0.5 * img_w, 0.5 * img_h])

        # Apply a random scaling to the src_rect.
        src_rect *= scale

        # Apply a random shift to the coordinates origin.
        src_rect[0::2] += sw * img_w
        src_rect[1::2] += sh * img_h

        # Map src_rect coordinates into image coordinates (center at corner).
        src_rect[0::2] += 0.5 * img_w
        src_rect[1::2] += 0.5 * img_h

        return ExtentTransform(
            src_rect=(src_rect[0], src_rect[1], src_rect[2], src_rect[3]),
            output_size=(int(src_rect[3] - src_rect[1]), int(src_rect[2] - src_rect[0])),
        )
    
    def _randomize_scale(self):
        return np.random.uniform(self.scale_range[0], self.scale_range[1])
    
    def _randomize_shift(self):
        sh = self.shift_range[1] * (np.random.rand() - 0.5)
        sw = self.shift_range[0] * (np.random.rand() - 0.5)
        return sh, sw


class RandomContrast(Augmentation):
    """
    Randomly transforms image contrast.

    Contrast intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce contrast
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase contrast

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min, intensity_max):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        super().__init__()
        self._init(locals())
        self._init_random_params(('w',))

    def randomize_parameters(self):
        self.set_random_param('w', np.random.uniform(self.intensity_min, self.intensity_max))

    def get_transform(self, img):
        w = self.get_random_param('w')
        if w is None:
            w = np.random.uniform(self.intensity_min, self.intensity_max)
        
        return BlendTransform(src_image=img.mean(), src_weight=1 - w, dst_weight=w)


class RandomBrightness(Augmentation):
    """
    Randomly transforms image brightness.

    Brightness intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce brightness
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase brightness

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min, intensity_max):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        super().__init__()
        self._init(locals())
        self._init_random_params(('w',))

    def randomize_parameters(self):
        self.set_random_param('w', np.random.uniform(self.intensity_min, self.intensity_max))

    def get_transform(self, img):
        w = self.get_random_param('w')
        if w is None:
            w = np.random.uniform(self.intensity_min, self.intensity_max)
        
        return BlendTransform(src_image=0, src_weight=1 - w, dst_weight=w)


class RandomSaturation(Augmentation):
    """
    Randomly transforms saturation of an RGB image.
    Input images are assumed to have 'RGB' channel order.

    Saturation intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce saturation (make the image more grayscale)
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase saturation

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min, intensity_max):
        """
        Args:
            intensity_min (float): Minimum augmentation (1 preserves input).
            intensity_max (float): Maximum augmentation (1 preserves input).
        """
        super().__init__()
        self._init(locals())
        self._init_random_params(('w',))

    def randomize_parameters(self):
        self.set_random_param('w', np.random.uniform(self.intensity_min, self.intensity_max))

    def get_transform(self, img):
        assert img.shape[-1] == 3, "RandomSaturation only works on RGB images"

        w = self.get_random_param('w')
        if w is None:
            w = np.random.uniform(self.intensity_min, self.intensity_max)
        
        grayscale = img.dot([0.299, 0.587, 0.114])[:, :, np.newaxis]
        return BlendTransform(src_image=grayscale, src_weight=1 - w, dst_weight=w)


class RandomLighting(Augmentation):
    """
    The "lighting" augmentation described in AlexNet, using fixed PCA over ImageNet.
    Input images are assumed to have 'RGB' channel order.

    The degree of color jittering is randomly sampled via a normal distribution,
    with standard deviation given by the scale parameter.
    """

    def __init__(self, scale):
        """
        Args:
            scale (float): Standard deviation of principal component weighting.
        """
        super().__init__()
        self._init(locals())
        self.eigen_vecs = np.array(
            [[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]
        )
        self.eigen_vals = np.array([0.2175, 0.0188, 0.0045])
        self._init_random_params(('weights',))

    def randomize_parameters(self):
        self.set_random_param('weights', np.random.normal(scale=self.scale, size=3))

    def get_transform(self, img):
        assert img.shape[-1] == 3, "RandomLighting only works on RGB images"

        weights = self.get_random_param('weights')
        if weights is None:
            weights = np.random.normal(scale=self.scale, size=3)
        
        return BlendTransform(
            src_image=self.eigen_vecs.dot(weights * self.eigen_vals), src_weight=1.0, dst_weight=1.0
        )
