import numpy as np
from numpy import random
import cv2
from PIL import Image, ImageEnhance, ImageOps

def rescale_pts(pts, down_ratio):
    if pts is None:
        return None
    return np.asarray(pts, np.float32)/float(down_ratio)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, pts):
        for t in self.transforms:
            img, pts = t(img, pts)
        return img, pts

class ConvertImgFloat(object):
    def __call__(self, img, pts):
        if pts is not None:
            pts = pts.astype(np.float32)
        return img.astype(np.float32), pts

class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, img, pts):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            img *= alpha
        return img, pts


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, img, pts):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            img += delta
        return img, pts

class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps
    def __call__(self, img):
        img = img[:, :, self.swaps]
        return img


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))
    def __call__(self, img, pts):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)
            img = shuffle(img)
        return img, pts


class PhotometricDistort(object):
    def __init__(self):
        self.pd = RandomContrast()
        self.rb = RandomBrightness()
        self.rln = RandomLightingNoise()

    def __call__(self, img, pts):
        img, pts = self.rb(img, pts)
        img, pts = self.pd(img, pts)
        img, pts = self.rln(img, pts)
        return img, pts


class Expand(object):
    def __init__(self, max_scale = 1.5, mean = (0.5, 0.5, 0.5)):
        self.mean = mean
        self.max_scale = max_scale

    def __call__(self, img, pts):
        if random.randint(2):
            return img, pts
        if pts is None:
            return img, pts
            
        h,w,c = img.shape
        ratio = random.uniform(1,self.max_scale)
        y1 = random.uniform(0, h*ratio-h)
        x1 = random.uniform(0, w*ratio-w)
        

        if len(pts) == 0 or np.max(pts[:,0])+int(x1)>w-1 or np.max(pts[:,1])+int(y1)>h-1:
            return img, pts
        else:
            expand_img = np.zeros(shape=(int(h*ratio), int(w*ratio),c),dtype=img.dtype)
            expand_img[:,:,:] = self.mean
            expand_img[int(y1):int(y1+h), int(x1):int(x1+w)] = img
            pts[:, 0] += int(x1)
            pts[:, 1] += int(y1)
            return expand_img, pts


class RandomSampleCrop(object):
    def __init__(self, ratio=(0.5, 1.5), min_win = 0.9):
        self.sample_options = ((None,), (0.7, None), (0.9, None), (None, None))
        self.ratio = ratio
        self.min_win = min_win

    def __call__(self, img, pts):
        if pts is None:
            return img, pts

        height, width ,_ = img.shape
        while True:
            mode = random.choice(self.sample_options)
            if mode is None:
                return img, pts
            for _ in range(50):
                current_img = img
                current_pts = pts.copy()
                w = random.uniform(self.min_win*width, width)
                h = random.uniform(self.min_win*height, height)
                if h/w<self.ratio[0] or h/w>self.ratio[1]:
                    continue
                y1 = random.uniform(height-h)
                x1 = random.uniform(width-w)
                rect = np.array([int(y1), int(x1), int(y1+h), int(x1+w)])
                current_img = current_img[rect[0]:rect[2], rect[1]:rect[3], :]
                current_pts[:, 0] -= rect[1]
                current_pts[:, 1] -= rect[0]
                pts_new = []
                for pt in current_pts:
                    if any(pt)<0 or pt[0]>current_img.shape[1]-1 or pt[1]>current_img.shape[0]-1:
                        continue
                    else:
                        pts_new.append(pt)
                if len(pts_new) == 0:
                    continue

                return current_img, np.asarray(pts_new, np.float32)

class RandomMirror_w(object):
    def __call__(self, img, pts):
        _,w,_ = img.shape
        if random.randint(2):
            img = img[:,::-1,:]
            if pts is not None:
                pts[:,0] = w-pts[:,0]
        return img, pts

class RandomMirror_h(object):
    def __call__(self, img, pts):
        h,_,_ = img.shape
        if random.randint(2):
            img = img[::-1,:,:]
            if pts is not None:
                pts[:,1] = h-pts[:,1]
        return img, pts


class Resize(object):
    def __init__(self, h, w):
        self.dsize = (w,h)
    def __call__(self, img, pts):
        img_resized = cv2.resize(img, dsize=self.dsize)
        if pts is not None:
            h,w,c = img.shape
            pts[:, 0] = pts[:, 0]/w*self.dsize[0]
            pts[:, 1] = pts[:, 1]/h*self.dsize[1]
            return img_resized, np.asarray(pts)
        else:
            return img_resized, None


class Equalize(object):
    def __call__(self, img, pts):
        img = Image.fromarray(np.uint8(img))
        if np.random.rand() < 0.3:
            img = ImageOps.equalize(img)
        img = np.array(img)
        return img, pts

class Solarize(object):
    def __call__(self, img, pts):
        img = Image.fromarray(np.uint8(img))
        if np.random.rand() < 0.3:
            magnitudes = np.linspace(0, 256, 11)
            img = ImageOps.solarize(img, random.uniform(magnitudes[3], magnitudes[4]))
        img = np.array(img)

        return img, pts


class Posterize(object):
    def __call__(self, img, pts):
        img = Image.fromarray(np.uint8(img))
        if np.random.rand() < 0.3:
            magnitudes = np.linspace(4, 8, 11)
            img = ImageOps.posterize(img, int(round(random.uniform(magnitudes[3], magnitudes[4]))))
        img = np.array(img)

        return img, pts


class Color(object):
    def __call__(self, img, pts):
        img = Image.fromarray(np.uint8(img))
        if np.random.rand() < 0.3:
            magnitudes = np.linspace(0.1, 1.9, 11)
            img = ImageEnhance.Color(img).enhance(random.uniform(magnitudes[3], magnitudes[4]))
        img = np.array(img)

        return img, pts


class Sharpness(object):
    def __call__(self, img, pts):
        img = Image.fromarray(np.uint8(img))
        if np.random.rand() < 0.3:
            magnitudes = np.linspace(0.1, 1.9, 11)
            img = ImageEnhance.Sharpness(img).enhance(random.uniform(magnitudes[3], magnitudes[4]))
        img = np.array(img)
        return img, pts
    
    
class RandomScale(object):
    """
    以图像中心为基准，对图像和点进行随机缩放。
    支持：
        - RandomScale((0.4, 1.0))
        - RandomScale(0.8)  # 会被当成 (0.8, 0.8)，即固定缩放
    """
    def __init__(self, scale_range=(0.4, 1.0), interpolation=cv2.INTER_LINEAR):
        if isinstance(scale_range, (int, float)):
            self.scale_range = (float(scale_range), float(scale_range))
        else:
            assert len(scale_range) == 2, "scale_range 必须是长度为 2 的 tuple/list 或一个标量"
            self.scale_range = (float(scale_range[0]), float(scale_range[1]))

        self.interpolation = interpolation

    def __call__(self, img, pts):
        scale = random.uniform(self.scale_range[0], self.scale_range[1])

        h, w, c = img.shape
        new_h, new_w = int(h * scale), int(w * scale)
        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=self.interpolation)
        new_canvas = np.zeros_like(img, dtype=img.dtype)

        if pts is None:
            paste_x = (w - new_w) // 2
            paste_y = (h - new_h) // 2
            crop_x = abs(min(0, paste_x))
            crop_y = abs(min(0, paste_y))
            paste_x = max(0, paste_x)
            paste_y = max(0, paste_y)

            scaled_h, scaled_w, _ = scaled_img.shape
            paste_w = min(scaled_w - crop_x, w - paste_x)
            paste_h = min(scaled_h - crop_y, h - paste_y)

            if paste_w > 0 and paste_h > 0:
                new_canvas[paste_y:paste_y+paste_h, paste_x:paste_x+paste_w] = \
                    scaled_img[crop_y:crop_y+paste_h, crop_x:crop_x+paste_w]
            return new_canvas, None

        pts = pts.copy().astype(np.float32)
        pts *= scale

        paste_x = (w - new_w) // 2
        paste_y = (h - new_h) // 2
        crop_x = abs(min(0, paste_x))
        crop_y = abs(min(0, paste_y))
        paste_x = max(0, paste_x)
        paste_y = max(0, paste_y)

        scaled_h, scaled_w, _ = scaled_img.shape
        paste_w = min(scaled_w - crop_x, w - paste_x)
        paste_h = min(scaled_h - crop_y, h - paste_y)

        if paste_w > 0 and paste_h > 0:
            new_canvas[paste_y:paste_y+paste_h, paste_x:paste_x+paste_w] = \
                scaled_img[crop_y:crop_y+paste_h, crop_x:crop_x+paste_w]
            pts[:, 0] += paste_x - crop_x
            pts[:, 1] += paste_y - crop_y

        return new_canvas, pts

    
class RandomRotate(object):
    """
    对图像和关键点进行随机旋转。
    """
    def __init__(self, angle_range=(-15, 15), prob=0.5):
        """
        Args:
            angle_range (tuple): 随机旋转的角度范围 (最小值, 最大值)，单位为度。
            prob (float): 执行此操作的概率。
        """
        self.angle_range = angle_range
        self.prob = prob

    def __call__(self, img, pts):
        # 1. 根据概率决定是否执行旋转
        if random.random() >= self.prob:
            return img, pts

        # 2. 获取图像中心和随机旋转角度
        h, w, _ = img.shape
        center = (w / 2, h / 2)
        angle = random.uniform(self.angle_range[0], self.angle_range[1])

        # 3. 计算OpenCV所需的旋转矩阵
        #    参数: 中心点, 角度, 缩放比例
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 4. 对图像进行仿射变换（旋转）
        rotated_img = cv2.warpAffine(img, M, (w, h))

        # 5. 对关键点进行同样的旋转变换
        if pts is not None and len(pts) > 0:
            # 创建一个(N, 3)的矩阵，其中N是点的数量
            # [x, y] -> [x, y, 1] 方便进行矩阵乘法
            pts_homogeneous = np.hstack([pts, np.ones((len(pts), 1))])
            
            # 使用旋转矩阵M对所有点进行变换
            # M (2x3) @ pts_homogeneous.T (3xN) -> transformed_pts (2xN)
            # 再转置回 (N, 2)
            rotated_pts = M.dot(pts_homogeneous.T).T
            return rotated_img, rotated_pts
        else:
            return rotated_img, pts
