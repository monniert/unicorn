import PIL
from torch.nn import functional as F
from torchvision.transforms import functional as Fvision

from utils.image import resize


class SquarePad:
    def __init__(self, padding_mode='constant'):
        super().__init__()
        self.padding_mode = padding_mode

    def __call__(self, image):
        if isinstance(image, PIL.Image.Image):
            max_wh = max(image.size)
            p_left, p_top = [(max_wh - s) // 2 for s in image.size]
            p_right, p_bottom = [max_wh - (s + pad) for s, pad in zip(image.size, [p_left, p_top])]
            padding = (p_left, p_top, p_right, p_bottom)
            return Fvision.pad(image, padding, fill=0, padding_mode=self.padding_mode)

        else:
            assert self.padding_mode == 'constant', 'padding_mode not supported for tensors, use "constant" instead'
            img_size = list(image.shape[-2:])
            max_wh = max(img_size)
            p_left, p_top = [(max_wh - s) // 2 for s in img_size]
            p_right, p_bottom = [max_wh - (s + pad) for s, pad in zip(img_size, [p_left, p_top])]
            padding = (p_top, p_bottom, p_left, p_right)
            return F.pad(image, padding, value=0, mode=self.padding_mode)


class Resize:
    def __init__(self, size, keep_aspect_ratio=True, resample=PIL.Image.ANTIALIAS, fit_inside=True):
        super().__init__()
        self.size = (size, size) if isinstance(size, int) else size
        self.keep_aspect_ratio = keep_aspect_ratio
        self.resample = resample
        self.fit_inside = fit_inside

    def __call__(self, img):
        return resize(img, self.size, keep_aspect_ratio=self.keep_aspect_ratio, resample=self.resample,
                      fit_inside=self.fit_inside)


class TensorResize():
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, img):
        return F.interpolate(img.unsqueeze(0), self.img_size, mode='bilinear', align_corners=False)[0]

    def __repr__(self):
        return self.__class__.__name__ + '()'


class TensorCenterCrop():
    def __init__(self, img_size):
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size

    def __call__(self, img):
        image_width, image_height = img.shape[-2:]
        height, width = self.img_size

        top = int((image_height - height + 1) * 0.5)
        left = int((image_width - width + 1) * 0.5)
        return img[..., top:top + height, left:left + width]

    def __repr__(self):
        return self.__class__.__name__ + '()'
