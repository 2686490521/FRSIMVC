"""Pixel-level missingness simulators for the pre-SLIC corruption pipeline.

Outline-v3 protocol
-------------------
Missingness is created on the 2-D image plane **before** SLIC, so that the
superpixel partition, the node features and the graph topology are all
derived from corrupted observations:

    Raw view -> Mask -> Observed-only preprocessing -> SLIC -> ...

Three protocols are provided through ``missing_mode``:

``random``  scattered small patches     (random-region masking)
``block``   compact rectangular / irregular occlusions
``strip``   long thin bands             (stripe failure / coverage gap)

All three share one implementation: a canvas is filled with axis-aligned
rectangles drawn from a protocol-specific size distribution.  ``random``
uses many small squares, ``block`` a few large rectangles with jagged
edges, ``strip`` full-span bands (periodically spaced by default).  The
last rectangle is trimmed on a shape-preserving axis so the canvas holds
*exactly* ``round(missing_rate * H * W)`` pixels.  This keeps the three
protocols on an identical missing-pixel budget, which is what makes them
comparable.

Nothing in this module may look at clean data: it only needs the image
shape, the rate and the seed.
"""

import math

import numpy as np

MODES = ('random', 'block', 'strip')

# Default geometry per protocol.  Scales are fractions of the image extent
# unless stated otherwise.
DEFAULT_PARAMS = {
    # scattered small patches: side ~ 3-7% of min(H, W)
    'random': {
        'patch_scale': (0.03, 0.07),
    },
    # large continuous occlusions + mildly jagged edges ("不规则区域")
    'block': {
        'block_scale_h': (0.10, 0.30),
        'block_scale_w': (0.10, 0.30),
        'irregularity': 0.25,
    },
    # long thin bands simulating stripe failure / coverage gaps
    'strip': {
        'strip_width_scale': (0.03, 0.09),
        'strip_orientation': 'auto',      # auto | vertical | horizontal
        'strip_layout': 'periodic',       # periodic | random
    },
}

# Rectangles thinner than this are replaced by a compact patch so the final
# residual never shows up as a one-pixel sliver in the mask visualisations.
MIN_VISIBLE_SIDE = 3


def _shrink_to(height, width, budget):
    """Largest rectangle <= (height, width) holding at most ``budget`` pixels."""
    if budget <= 0:
        return 0, 0
    if height * width <= budget:
        return height, width
    scale = math.sqrt(budget / float(height * width))
    rows = max(1, min(height, int(height * scale)))
    cols = max(1, min(width, budget // rows))
    return rows, cols


def _compact_shape(budget):
    """Square-ish rectangle holding at most ``budget`` pixels."""
    budget = max(1, int(budget))
    side = max(1, int(math.sqrt(budget)))
    return side, max(1, min(side, budget // side))


def _sample_size(rng, mode, params, height, width, orientation):
    """Draw one rectangle (rows, cols) for the requested protocol."""
    if mode == 'random':
        low, high = params['patch_scale']
        side = float(min(height, width))
        return rng.uniform(low, high) * side, rng.uniform(low, high) * side
    if mode == 'block':
        h_low, h_high = params['block_scale_h']
        w_low, w_high = params['block_scale_w']
        return rng.uniform(h_low, h_high) * height, rng.uniform(w_low, w_high) * width
    if mode == 'strip':
        low, high = params['strip_width_scale']
        band = float(rng.uniform(low, high))
        if orientation == 'vertical':
            return float(height), band * width
        return band * height, float(width)
    raise ValueError(f'Unsupported missing_mode {mode!r}; expected one of {MODES}')


def _draw(canvas, top, left, rows, cols, rng, mode, params):
    if mode == 'block' and float(params.get('irregularity', 0.0)) > 0.0:
        # Jagged left/right edges.  Every row keeps the same width, so the
        # pixel budget is untouched but the region stops being a clean
        # rectangle ("连续矩形或者不规则区域").
        height, width = canvas.shape
        reference = max(1, min(rows, cols))
        max_shift = max(1, int(round(float(params['irregularity']) * reference * 0.5)))
        offsets = rng.integers(-max_shift, max_shift + 1, size=rows)
        for row in range(rows):
            start = int(np.clip(left + offsets[row], 0, width - cols))
            canvas[top + row, start:start + cols] = True
    else:
        canvas[top:top + rows, left:left + cols] = True


def _periodic_state(rng, params, target, height, width, orientation):
    """Cursor + spacing for a periodic band layout covering the whole axis.

    The spacing is adapted to the requested budget so the bands are spread
    over the full extent instead of saturating one end of the image.
    """
    if orientation == 'vertical':
        axis, span = float(width), float(height)
        band_mean = 0.5 * (float(params['strip_width_scale'][0]) + float(params['strip_width_scale'][1])) * axis
    else:
        axis, span = float(height), float(width)
        band_mean = 0.5 * (float(params['strip_width_scale'][0]) + float(params['strip_width_scale'][1])) * axis
    band_mean = max(1.0, band_mean)
    n_bands = max(2, int(round((float(target) / span) / band_mean)))
    step = max(1.0, axis / float(n_bands))
    return float(rng.random()) * axis, step


def _place_rectangles(canvas, target, rng, mode, params, orientation=None):
    """Greedily fill ``canvas`` with rectangles until it holds ``target`` pixels."""
    height, width = canvas.shape
    filled = int(canvas.sum())
    periodic = (
        mode == 'strip'
        and str(params.get('strip_layout', 'periodic')).lower() == 'periodic'
    )
    if periodic:
        cursor, step = _periodic_state(rng, params, target, height, width, orientation)
        axis_len = float(width if orientation == 'vertical' else height)
        position = 'periodic'
    else:
        cursor = step = axis_len = 0.0
        position = 'random'

    attempts = 0
    stagnant = 0
    # Only reached with a pathological draw distribution (a tiny target inside
    # already-covered area); keeps the loop bounded.
    max_attempts = 20000
    while filled < target and attempts < max_attempts:
        attempts += 1
        remaining = target - filled
        rows, cols = _sample_size(rng, mode, params, height, width, orientation)
        rows = int(np.clip(round(rows), 1, height))
        cols = int(np.clip(round(cols), 1, width))
        if rows * cols > remaining:
            # Shape-preserving trim so the pixel count lands exactly on the
            # requested rate instead of overshooting.
            rows, cols = _shrink_to(rows, cols, remaining)
            if rows < 1 or cols < 1:
                break
            if min(rows, cols) < MIN_VISIBLE_SIDE:
                rows, cols = _compact_shape(remaining)
        top = int(rng.integers(0, height - rows + 1))
        left = int(rng.integers(0, width - cols + 1))
        if position == 'periodic':
            if orientation == 'vertical':
                left = int(np.clip(round(cursor), 0, width - cols))
            else:
                top = int(np.clip(round(cursor), 0, height - rows))
            cursor = (cursor + step) % axis_len
        _draw(canvas, top, left, rows, cols, rng, mode, params)
        updated = int(canvas.sum())
        if updated <= filled:
            # The periodic cursor ran out of fresh positions (or the draw
            # landed on covered pixels): stop wasting attempts on it.
            stagnant += 1
            if stagnant >= 40:
                position = 'random'
                stagnant = 0
        else:
            stagnant = 0
        filled = updated
    return canvas, attempts


def generate_pixel_masks(
    n_views,
    height,
    width,
    missing_rate,
    seed,
    mode='random',
    params=None,
):
    """Return ``(n_views, height, width)`` boolean masks.

    ``True`` means *observed*, ``False`` means *missing/corrupted* -- the same
    polarity as the previous node-level ``observation_masks``, so downstream
    code can treat it as ``M^(v)`` in ``X_corrupt = M ⊙ X``.
    """
    mode = str(mode).lower()
    if mode not in MODES:
        raise ValueError(f'Unknown missing_mode {mode!r}; expected one of {MODES}')
    missing_rate = float(missing_rate)
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError(f'missing_rate must be in [0, 1), got {missing_rate}')

    height = int(height)
    width = int(width)
    merged = dict(DEFAULT_PARAMS[mode])
    if params:
        merged.update({key: value for key, value in params.items() if value is not None})

    rng = np.random.default_rng(int(seed))
    target = int(round(missing_rate * height * width))
    masks = np.ones((int(n_views), height, width), dtype=bool)
    if target <= 0:
        return masks

    for view in range(int(n_views)):
        orientation = None
        if mode == 'strip':
            orientation = str(merged.get('strip_orientation', 'auto')).lower()
            if orientation == 'auto':
                orientation = 'vertical' if rng.random() < 0.5 else 'horizontal'
            if orientation not in ('vertical', 'horizontal'):
                raise ValueError(
                    f'strip_orientation must be auto|vertical|horizontal, got {orientation!r}'
                )
        canvas = np.zeros((height, width), dtype=bool)
        canvas, _ = _place_rectangles(canvas, target, rng, mode, merged, orientation)
        masks[view] = ~canvas
    return masks


def node_observed_ratios(pixel_observed, sp_labels, n_nodes=None):
    """``r_i^(v) = |P_i ∩ observed| / |P_i|`` for every superpixel node."""
    labels = np.asarray(sp_labels).reshape(-1).astype(np.int64)
    if n_nodes is None:
        n_nodes = int(labels.max()) + 1
    n_nodes = int(n_nodes)
    counts = np.bincount(labels, minlength=n_nodes).astype(np.float64)
    observed = np.bincount(
        labels,
        weights=np.asarray(pixel_observed, dtype=np.float64).reshape(-1),
        minlength=n_nodes,
    )
    return (observed / np.maximum(counts, 1.0)).astype(np.float64)


def derive_node_masks(ratios, threshold):
    """``m_i^(v) = 1[ r_i^(v) >= tau ]``; input/output shape ``(V, N)``."""
    return np.asarray(ratios, dtype=np.float64) >= float(threshold)


def repair_all_missing_nodes(node_masks, ratios):
    """Keep at least one modality per node, as in the paper setting.

    Nodes missing in every view are restored in the view with the largest
    observed ratio.  Returns ``(masks, n_repaired)``.
    """
    masks = np.asarray(node_masks).copy()
    if masks.ndim != 2 or masks.shape[0] < 2:
        return masks, 0
    all_missing = ~masks.any(axis=0)
    n_repaired = int(all_missing.sum())
    if n_repaired:
        best_view = np.argmax(np.asarray(ratios, dtype=np.float64)[:, all_missing], axis=0)
        masks[best_view, np.flatnonzero(all_missing)] = True
    return masks, n_repaired


def summarize_pixel_masks(pixel_masks):
    """Per-view pixel-level statistics (actual rate, missing pixel count)."""
    pixel_masks = np.asarray(pixel_masks, dtype=bool)
    return [
        {
            'view': int(view),
            'missing_pixels': int((~pixel_masks[view]).sum()),
            'pixel_missing_rate': float((~pixel_masks[view]).mean()),
        }
        for view in range(pixel_masks.shape[0])
    ]
