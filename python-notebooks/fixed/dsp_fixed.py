"""
dsp_fixed: fixed-point Q15 arithmetic using NumPy + Numba.

Q15 convention:
    real = raw / 2**15
    raw is normally int16 for stored values.
Products use an int32 intermediate and are shifted right by 15 bits.
Convolution uses int32 accumulation and returns int32 raw Q15 values.
"""

import numpy as np
from numba import njit

FRAC_BITS = 15
SCALE = 1 << FRAC_BITS


def float_to_q15(x):
    """Convert scalar or array of floats to Q15 int16 with saturation."""
    x = np.asarray(x, dtype=np.float64)
    raw = np.rint(x * SCALE)
    raw = np.clip(raw, -32768, 32767)
    return raw.astype(np.int16)


def q15_to_float(x):
    """Convert Q15 raw scalar/array to float64."""
    return np.asarray(x, dtype=np.float64) / SCALE


@njit(cache=True)
def q15_add(a, b):
    """Q15 scalar addition, with int16 saturation."""
    s = int(a) + int(b)
    if s > 32767:
        s = 32767
    elif s < -32768:
        s = -32768
    return np.int16(s)


@njit(cache=True)
def q15_sub(a, b):
    """Q15 scalar subtraction, with int16 saturation."""
    s = int(a) - int(b)
    if s > 32767:
        s = 32767
    elif s < -32768:
        s = -32768
    return np.int16(s)


@njit(cache=True)
def q15_mul(a, b):
    """Q15 scalar multiplication, with rounding and int16 saturation."""
    p = int(a) * int(b)
    # Symmetric round-to-nearest before scaling back.
    if p >= 0:
        p += 1 << (FRAC_BITS - 1)
    else:
        p -= 1 << (FRAC_BITS - 1)
    y = p >> FRAC_BITS
    if y > 32767:
        y = 32767
    elif y < -32768:
        y = -32768
    return np.int16(y)


@njit(cache=True)
def q15_mac(acc, a, b):
    """Q15 multiply-accumulate: acc + a*b, saturated to int16."""
    p = int(a) * int(b)
    if p >= 0:
        p += 1 << (FRAC_BITS - 1)
    else:
        p -= 1 << (FRAC_BITS - 1)
    product = p >> FRAC_BITS
    y = int(acc) + product
    if y > 32767:
        y = 32767
    elif y < -32768:
        y = -32768
    return np.int16(y)


@njit(cache=True)
def q15_convolve(x, h):
    """
    Linear convolution of two Q15 arrays.
    Accumulator is int64 to avoid overflow during the sum of products.
    Output is int32 raw Q15.
    """
    n = len(x)
    m = len(h)
    y = np.zeros(n + m - 1, dtype=np.int32)

    for i in range(n):
        xi = int(x[i])
        for j in range(m):
            y[i + j] += xi * int(h[j])

    # Product of two Q15 numbers has 30 fractional bits;
    # convert accumulated result back to Q15.
    for k in range(len(y)):
        p = int(y[k])
        if p >= 0:
            p += 1 << (FRAC_BITS - 1)
        else:
            p -= 1 << (FRAC_BITS - 1)
        y[k] = p >> FRAC_BITS

    return y
