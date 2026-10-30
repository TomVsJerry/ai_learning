"""用训好的权重读一张图片里的一个数字。

MNIST 的图是黑底、白字、居中、28×28、亮度 0～1。
手机拍的一般是白纸黑字，还可能带着横格线。
这里先把照片收成同一张样子，再交给 DigitClassifier。

先训练，权重会写到本目录的 digit.pt：

    python train.py --epochs 2

再读一张图：

    python predict.py photo.jpg
    python predict.py photo.jpg --weights digit.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import DigitClassifier


DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "digit.pt"
# 比纸面暗这么多，才当成钢笔字。横格线和灰纸更浅，会被收成黑色背景。
INK_DARKER_THAN_PAPER = 37


def paper_level(gray: np.ndarray) -> int:
    """纸面是画面里最常见的亮度。过曝的白边另算，不拿来当纸。"""
    hist = np.bincount(np.clip(gray, 0, 255).astype(np.uint8).ravel(), minlength=256)
    if hist[250:].sum() > 0.15 * gray.size:
        hist = hist.copy()
        hist[250:] = 0
    return int(np.argmax(hist))


def max_vertical_run(mask: np.ndarray) -> np.ndarray:
    """每一列上最长的一段连续笔画。分开的几条横线不会加在一起。"""
    best = np.zeros(mask.shape[1], dtype=np.int32)
    run = np.zeros(mask.shape[1], dtype=np.int32)
    for row in mask:
        run = np.where(row, run + 1, 0)
        best = np.maximum(best, run)
    return best


def digit_column_span(runs: np.ndarray):
    """字身是比较高的那些列。贴着纸边、只有几列宽的阴影不算进字里。"""
    peak = int(runs.max()) if runs.size else 0
    if peak < 6:
        return None
    cols = np.flatnonzero(runs >= max(5, int(0.35 * peak)))
    if cols.size == 0:
        return None

    groups = []
    start = previous = int(cols[0])
    for x in cols.tolist()[1:]:
        if x - previous > 12:
            groups.append((start, previous))
            start = x
        previous = x
    groups.append((start, previous))

    width = int(runs.size)
    kept = []
    for group_start, group_end in groups:
        on_edge = group_start <= 2 or group_end >= width - 3
        if on_edge and group_end - group_start <= 4:
            continue
        kept.append((group_start, group_end))
    if not kept:
        kept = groups
    left = min(group_start for group_start, _group_end in kept)
    right = max(group_end for _group_start, group_end in kept)
    return left, right


def remove_ruled_lines(mask: np.ndarray) -> np.ndarray:
    """横格线又扁又长，会从字的两侧伸出去。和笔画交叉的那一段留下。"""
    runs = max_vertical_run(mask)
    span = digit_column_span(runs)
    if span is None:
        return mask
    left, right = span
    peak = int(runs.max())
    line_thickness = max(4, int(0.22 * peak))
    digit_width = right - left + 1
    height = mask.shape[0]
    cleaned = mask.copy()

    for y in range(height):
        cols = np.flatnonzero(mask[y])
        if cols.size == 0:
            continue
        outside = np.count_nonzero((cols < left - 3) | (cols > right + 3))
        row_span = int(cols[-1] - cols[0] + 1)
        if outside < 8 and row_span < digit_width * 1.5:
            continue
        for x in cols.tolist():
            y0 = y
            while y0 > 0 and mask[y0 - 1, x]:
                y0 -= 1
            y1 = y
            while y1 + 1 < height and mask[y1 + 1, x]:
                y1 += 1
            if y1 - y0 + 1 <= line_thickness:
                cleaned[y, x] = False
    return cleaned


def largest_component(mask: np.ndarray) -> np.ndarray:
    """只留最大的一块笔画，散落的横线碎片和纸边污点去掉。"""
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    best = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        pixels = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(pixels) > len(best):
            best = pixels

    kept = np.zeros_like(mask)
    if len(best) >= 12:
        yy, xx = zip(*best)
        kept[list(yy), list(xx)] = True
    return kept


def dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """钢笔字比 MNIST 的笔画细，加粗一圈再送进网络。"""
    height, width = mask.shape
    thickened = np.zeros_like(mask)
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        thickened[
            max(0, y - radius) : min(height, y + radius + 1),
            max(0, x - radius) : min(width, x + radius + 1),
        ] = True
    return thickened


def fit_canvas(ink: np.ndarray) -> np.ndarray:
    """裁到笔画外框，最长边收到 20，再把重心挪到 28×28 的正中。"""
    ys, xs = np.nonzero(ink > 0.05)
    if xs.size == 0:
        return np.zeros((28, 28), dtype=np.float32)

    crop = ink[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]
    crop_height, crop_width = crop.shape
    scale = 20 / max(crop_height, crop_width)
    resized_height = max(1, int(round(crop_height * scale)))
    resized_width = max(1, int(round(crop_width * scale)))
    resized = np.asarray(
        Image.fromarray((np.clip(crop, 0, 1) * 255).astype(np.uint8), mode="L").resize(
            (resized_width, resized_height), Image.Resampling.LANCZOS
        ),
        dtype=np.float32,
    )
    resized /= 255

    canvas = np.zeros((28, 28), dtype=np.float32)
    top = (28 - resized_height) // 2
    left = (28 - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized

    total = float(canvas.sum())
    if total <= 0:
        return canvas
    yy, xx = np.mgrid[0:28, 0:28]
    shift_y = int(round(14 - float((yy * canvas).sum()) / total))
    shift_x = int(round(14 - float((xx * canvas).sum()) / total))
    shifted = np.zeros_like(canvas)
    src_y0 = max(0, -shift_y)
    src_x0 = max(0, -shift_x)
    dst_y0 = max(0, shift_y)
    dst_x0 = max(0, shift_x)
    copy_height = min(28 - src_y0, 28 - dst_y0)
    copy_width = min(28 - src_x0, 28 - dst_x0)
    shifted[dst_y0 : dst_y0 + copy_height, dst_x0 : dst_x0 + copy_width] = canvas[
        src_y0 : src_y0 + copy_height, src_x0 : src_x0 + copy_width
    ]
    return shifted


def ink_on_paper(gray: np.ndarray) -> np.ndarray:
    """白纸黑字：纸面收成 0，笔画收成 1，再去掉横格线。"""
    paper = paper_level(gray)
    darkness = np.clip(paper - gray, 0, None)
    mask = darkness > INK_DARKER_THAN_PAPER
    mask = remove_ruled_lines(mask)
    mask = largest_component(mask)
    mask = dilate(mask)
    return fit_canvas(mask.astype(np.float32))


def digit_on_black(gray: np.ndarray) -> np.ndarray:
    """已经是黑底白字时，不再按白纸来反色。很小的图保持原样缩到 28×28。"""
    if max(gray.shape) <= 40:
        image = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8), mode="L")
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255

    peak = float(np.percentile(gray, 99.5))
    if peak < 1:
        return np.zeros((28, 28), dtype=np.float32)
    mask = gray > max(20, 0.3 * peak)
    mask = largest_component(mask)
    mask = dilate(mask)
    return fit_canvas(mask.astype(np.float32))


def load_digit(path: Path) -> torch.Tensor:
    """把一张图收成 (1, 1, 28, 28)，亮度和训练集一样落在 0～1。"""
    gray = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    if paper_level(gray) < 80:
        canvas = digit_on_black(gray)
    else:
        canvas = ink_on_paper(gray)
    return torch.from_numpy(canvas).view(1, 1, 28, 28)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="用训好的权重读一张图片")
    parser.add_argument("image", type=Path)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--hidden-size", type=int, default=128)
    args = parser.parse_args()

    model = DigitClassifier(args.hidden_size)
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    model.eval()

    scores = model(load_digit(args.image))
    probabilities = torch.softmax(scores, dim=1)[0]
    predicted = int(probabilities.argmax())

    print(f"读成 {predicted}")
    for digit, probability in enumerate(probabilities.tolist()):
        print(f"  {digit}: {probability:.4f}")


if __name__ == "__main__":
    main()
