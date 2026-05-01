"""
远光灯场景图像预处理模块
针对夜间行车远光灯导致的过曝、眩光问题进行图像增强
"""

import cv2
import numpy as np


# ──────────────────────────────────────────────
# 1. 基础亮度 / 对比度校正
# ──────────────────────────────────────────────

def gamma_correction(img: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    """
    Gamma 校正：gamma < 1 压暗高光区域，缓解过曝。
    """
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8
    )
    return cv2.LUT(img, table)


def clahe_enhance(img: np.ndarray,
                  clip_limit: float = 2.0,
                  tile_grid: tuple = (8, 8)) -> np.ndarray:
    """
    CLAHE（限制对比度自适应直方图均衡化）：
    在 LAB 色彩空间的 L 通道上操作，保留色彩信息的同时增强暗部细节。
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ──────────────────────────────────────────────
# 2. 眩光（Glare）抑制
# ──────────────────────────────────────────────

def detect_glare_mask(img: np.ndarray, threshold: int = 240) -> np.ndarray:
    """
    检测过曝（眩光）区域，返回二值掩码（255 = 眩光区域）。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    # 膨胀掩码以覆盖光晕边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def inpaint_glare(img: np.ndarray, glare_mask: np.ndarray) -> np.ndarray:
    """
    使用 Telea 算法对眩光区域进行图像修复（inpainting）。
    """
    return cv2.inpaint(img, glare_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)


def suppress_glare(img: np.ndarray, threshold: int = 240) -> np.ndarray:
    """一键眩光抑制：检测 + 修复。"""
    mask = detect_glare_mask(img, threshold)
    if mask.sum() == 0:
        return img
    return inpaint_glare(img, mask)


# ──────────────────────────────────────────────
# 3. 多尺度 Retinex（MSR）—— 模拟人眼对强光的适应
# ──────────────────────────────────────────────

def _single_scale_retinex(img_float: np.ndarray, sigma: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(img_float, (0, 0), sigma)
    blurred = np.where(blurred == 0, 1e-6, blurred)
    return np.log10(img_float + 1e-6) - np.log10(blurred)


def multi_scale_retinex(img: np.ndarray,
                        sigmas: list = (15, 80, 250),
                        weights: list = None) -> np.ndarray:
    """
    多尺度 Retinex：均衡大动态范围图像中的亮暗区域。
    适合处理远光灯造成的强烈明暗对比。
    """
    if weights is None:
        weights = [1.0 / len(sigmas)] * len(sigmas)

    img_float = img.astype(np.float32) + 1.0
    channels = cv2.split(img_float)
    result_channels = []

    for ch in channels:
        retinex = np.zeros_like(ch)
        for sigma, w in zip(sigmas, weights):
            retinex += w * _single_scale_retinex(ch, sigma)
        # 归一化到 [0, 255]
        retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min() + 1e-6)
        result_channels.append((retinex * 255).astype(np.uint8))

    return cv2.merge(result_channels)


# ──────────────────────────────────────────────
# 4. 双边滤波去噪（保留边缘）
# ──────────────────────────────────────────────

def bilateral_denoise(img: np.ndarray,
                      d: int = 9,
                      sigma_color: float = 75,
                      sigma_space: float = 75) -> np.ndarray:
    """双边滤波：去除噪声同时保留行人轮廓边缘。"""
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


# ──────────────────────────────────────────────
# 5. 综合预处理流水线
# ──────────────────────────────────────────────

class HighBeamPreprocessor:
    """
    远光灯场景行人检测预处理流水线。

    推荐配置（可在初始化时调整）：
      - glare_suppression: 先抑制眩光（防止后续操作受过曝影响）
      - retinex:           多尺度 Retinex 均衡明暗
      - clahe:             CLAHE 进一步增强暗部对比度
      - denoise:           双边滤波降噪
    """

    def __init__(
        self,
        use_glare_suppression: bool = True,
        glare_threshold: int = 240,
        use_retinex: bool = True,
        retinex_sigmas: list = None,
        use_clahe: bool = True,
        clahe_clip: float = 2.0,
        use_denoise: bool = False,
        use_gamma: bool = False,
        gamma: float = 0.7,
    ):
        self.use_glare_suppression = use_glare_suppression
        self.glare_threshold = glare_threshold
        self.use_retinex = use_retinex
        self.retinex_sigmas = retinex_sigmas or [15, 80, 250]
        self.use_clahe = use_clahe
        self.clahe_clip = clahe_clip
        self.use_denoise = use_denoise
        self.use_gamma = use_gamma
        self.gamma = gamma

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.process(img)

    def process(self, img: np.ndarray) -> np.ndarray:
        """对输入 BGR 图像执行完整预处理流水线，返回增强后的 BGR 图像。"""
        if img is None or img.size == 0:
            raise ValueError("输入图像为空")

        result = img.copy()

        if self.use_glare_suppression:
            result = suppress_glare(result, self.glare_threshold)

        if self.use_gamma:
            result = gamma_correction(result, self.gamma)

        if self.use_retinex:
            result = multi_scale_retinex(result, self.retinex_sigmas)

        if self.use_clahe:
            result = clahe_enhance(result, self.clahe_clip)

        if self.use_denoise:
            result = bilateral_denoise(result)

        return result

    def visualize_steps(self, img: np.ndarray) -> dict:
        """
        返回每个预处理步骤的中间结果，便于调试与论文图表生成。
        """
        steps = {"original": img.copy()}
        current = img.copy()

        if self.use_glare_suppression:
            current = suppress_glare(current, self.glare_threshold)
            steps["glare_suppressed"] = current.copy()

        if self.use_gamma:
            current = gamma_correction(current, self.gamma)
            steps["gamma_corrected"] = current.copy()

        if self.use_retinex:
            current = multi_scale_retinex(current, self.retinex_sigmas)
            steps["retinex"] = current.copy()

        if self.use_clahe:
            current = clahe_enhance(current, self.clahe_clip)
            steps["clahe"] = current.copy()

        if self.use_denoise:
            current = bilateral_denoise(current)
            steps["denoised"] = current.copy()

        steps["final"] = current
        return steps
