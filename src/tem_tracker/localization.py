from __future__ import annotations

import cv2
import numpy as np


def preprocess(frame: np.ndarray, config: dict) -> np.ndarray:
    im = frame.copy()
    if config.get("clahe", False):
        im = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(im)
    sigma = config.get("gaussian_sigma", 2)
    return cv2.GaussianBlur(im.astype(np.float32), (0, 0), sigma) if sigma > 0 else im.astype(np.float32)


def patch(image: np.ndarray, center: np.ndarray, radius: int) -> np.ndarray:
    return cv2.getRectSubPix(image.astype(np.float32), (2*radius+1, 2*radius+1), tuple(map(float, center)))


def disk(radius: int, fraction: float = 1.0) -> np.ndarray:
    y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
    return (x*x+y*y <= (radius*fraction)**2).astype(np.float32)


def masked_ncc(search: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Zero-mean circular-mask NCC, computed using OpenCV correlations."""
    r = template.shape[0]//2
    mask = disk(r, .95)
    n = mask.sum()
    centered = (template-float((template*mask).sum()/n))*mask
    tv = float((centered**2).sum())
    if tv < 1e-6:
        return np.full((search.shape[0]-2*r, search.shape[1]-2*r), -1, np.float32)
    num = cv2.matchTemplate(search, centered, cv2.TM_CCORR)
    sums = cv2.matchTemplate(search, mask, cv2.TM_CCORR)
    sums2 = cv2.matchTemplate(search*search, mask, cv2.TM_CCORR)
    var = np.maximum(sums2-sums*sums/n, 0)
    return np.clip(num/np.sqrt(np.maximum(var*tv, 1e-6)), -1, 1)


def locate_template(image: np.ndarray, template: np.ndarray, expected: np.ndarray,
                    search_range: float) -> tuple[np.ndarray, float, float]:
    r = template.shape[0]//2
    h, w = image.shape
    # Border replication is only for correlation support. Candidate centers remain inside the real image.
    padded = cv2.copyMakeBorder(image, r, r, r, r, cv2.BORDER_REPLICATE)
    xmin = max(0, int(np.floor(expected[0]-search_range)))
    xmax = min(w-1, int(np.ceil(expected[0]+search_range)))
    ymin = max(0, int(np.floor(expected[1]-search_range)))
    ymax = min(h-1, int(np.ceil(expected[1]+search_range)))
    if xmin > xmax or ymin > ymax:
        return expected.copy(), -1., 0.
    search = padded[ymin:ymax+2*r+1, xmin:xmax+2*r+1]
    scores = masked_ncc(search, template)
    yy, xx = np.mgrid[ymin:ymax+1, xmin:xmax+1]
    scores[(xx-expected[0])**2+(yy-expected[1])**2 > search_range**2] = -1
    py, px = np.unravel_index(np.argmax(scores), scores.shape)
    score = float(scores[py, px])
    pos = np.array([xmin+px, ymin+py], dtype=float)
    for axis, (p, lim) in enumerate(((px, scores.shape[1]), (py, scores.shape[0]))):
        if 0 < p < lim-1:
            a, b, c = (scores[py, px-1:px+2] if axis == 0 else scores[py-1:py+2, px])
            denom = float(a-2*b+c)
            if denom < -1e-6:
                pos[axis] += float(np.clip(.5*(a-c)/denom, -.75, .75))
    other = scores.copy()
    gy, gx = np.ogrid[:scores.shape[0], :scores.shape[1]]
    other[(gx-px)**2+(gy-py)**2 < max(5, .65*r)**2] = -1
    margin = score-float(other.max())
    return pos, score, margin


def dark_center(image: np.ndarray, center: np.ndarray, radius: float,
                fraction: float = .75, max_shift: float = 6.) -> tuple[np.ndarray, float]:
    """Intensity centroid in a constrained disk; not a segmentation mask or geometric area."""
    r = max(4, int(round(radius)))
    crop = patch(image, center, r)
    y, x = np.mgrid[-r:r+1, -r:r+1]
    d = np.sqrt(x*x+y*y)
    ring = crop[(d > .80*r) & (d < r)]
    bg = float(np.percentile(ring, 75))
    contrast = bg-float(np.percentile(crop[d < .45*r], 25))
    weights = np.maximum(bg-crop, 0)*(d <= fraction*r)
    if weights.sum() < 1e-6:
        return center.copy(), contrast
    offset = np.array([(weights*x).sum(), (weights*y).sum()])/weights.sum()
    norm = np.linalg.norm(offset)
    if norm > max_shift:
        offset *= max_shift/norm
    return center+offset, contrast


def optical_step(prev: np.ndarray, current: np.ndarray, point: np.ndarray, config: dict):
    p = np.asarray(point, np.float32).reshape(1, 1, 2)
    a, b = np.clip(prev,0,255).astype(np.uint8), np.clip(current,0,255).astype(np.uint8)
    params = dict(winSize=(int(config["lk_window"]),)*2, maxLevel=int(config["lk_max_level"]),
                  criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 40, .01))
    q, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p, None, **params)
    if q is None or not st[0,0]:
        return point.copy(), float("inf")
    back, stb, _ = cv2.calcOpticalFlowPyrLK(b, a, q, None, **params)
    error = float(np.linalg.norm(back-p)) if stb[0,0] else float("inf")
    return q[0,0].astype(float), error
