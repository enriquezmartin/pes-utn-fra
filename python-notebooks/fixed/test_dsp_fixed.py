import numpy as np
import timeit

from dsp_fixed import (
    float_to_q15, q15_to_float,
    q15_add, q15_sub, q15_mul, q15_mac, q15_convolve
)

np.set_printoptions(precision=6, suppress=True)

# ------------------------------------------------------------
# 1. Arrays de prueba
# ------------------------------------------------------------

x_float = np.array([0.25, 0.5, -0.25, 0.75], dtype=np.float64)
h_float = np.array([0.5, -0.25, 0.125], dtype=np.float64)

x_fixed = float_to_q15(x_float)
h_fixed = float_to_q15(h_float)

print("x float :", x_float)
print("x Q15   :", x_fixed)
print("x back  :", q15_to_float(x_fixed))
print()
print("h float :", h_float)
print("h Q15   :", h_fixed)
print("h back  :", q15_to_float(h_fixed))
print()

# ------------------------------------------------------------
# 2. Operaciones escalares
# ------------------------------------------------------------

a = float_to_q15(0.5)
b = float_to_q15(0.25)
acc = float_to_q15(0.125)

print("=== Operaciones Q15 ===")
print("0.5 + 0.25 =", q15_to_float(q15_add(a, b)))
print("0.5 - 0.25 =", q15_to_float(q15_sub(a, b)))
print("0.5 * 0.25 =", q15_to_float(q15_mul(a, b)))
print("0.125 + 0.5*0.25 =", q15_to_float(q15_mac(acc, a, b)))
print()

# ------------------------------------------------------------
# 3. Convolución
# ------------------------------------------------------------

y_float = np.convolve(x_float, h_float)
y_fixed_raw = q15_convolve(x_fixed, h_fixed)
y_fixed = q15_to_float(y_fixed_raw)

print("=== Convolución ===")
print("float :", y_float)
print("fixed :", y_fixed)
print("error :", y_fixed - y_float)
print("max |error| =", np.max(np.abs(y_fixed - y_float)))
print()

# ------------------------------------------------------------
# 4. Benchmarks
#
# La primera llamada a una función Numba compila la función.
# La excluimos del benchmark haciendo una llamada previa.
# ------------------------------------------------------------

rng = np.random.default_rng(1234)
N = 1024
M = 64

x_float_b = rng.uniform(-0.9, 0.9, N).astype(np.float64)
h_float_b = rng.uniform(-0.5, 0.5, M).astype(np.float64)
x_fixed_b = float_to_q15(x_float_b)
h_fixed_b = float_to_q15(h_float_b)

# Warm-up / compilación Numba
q15_convolve(x_fixed_b, h_fixed_b)
q15_add(x_fixed_b[0], h_fixed_b[0])
q15_mul(x_fixed_b[0], h_fixed_b[0])
q15_mac(np.int16(0), x_fixed_b[0], h_fixed_b[0])

reps_conv = 30

t_float = timeit.timeit(
    lambda: np.convolve(x_float_b, h_float_b),
    number=reps_conv
) / reps_conv

t_fixed = timeit.timeit(
    lambda: q15_convolve(x_fixed_b, h_fixed_b),
    number=reps_conv
) / reps_conv

print("=== Benchmark convolución ===")
print(f"Tamaño: x={N}, h={M}")
print(f"float64 np.convolve : {t_float*1e3:.3f} ms")
print(f"Q15 + Numba         : {t_fixed*1e3:.3f} ms")
print(f"ratio fixed/float   : {t_fixed/t_float:.2f}x")
print()

# También comparamos float32
x_float32 = x_float_b.astype(np.float32)
h_float32 = h_float_b.astype(np.float32)

t_float32 = timeit.timeit(
    lambda: np.convolve(x_float32, h_float32),
    number=reps_conv
) / reps_conv

print(f"float32 np.convolve  : {t_float32*1e3:.3f} ms")
print(f"ratio Q15/float32    : {t_fixed/t_float32:.2f}x")
print()

# Benchmark de operaciones escalares.
# Es orientativo: Python sigue llamando funciones para cada operación.
reps_scalar = 1_000_000

tf_add = timeit.timeit(lambda: 0.5 + 0.25, number=reps_scalar)
tq_add = timeit.timeit(lambda: q15_add(a, b), number=reps_scalar)

tf_mul = timeit.timeit(lambda: 0.5 * 0.25, number=reps_scalar)
tq_mul = timeit.timeit(lambda: q15_mul(a, b), number=reps_scalar)

print("=== Benchmark escalar (orientativo) ===")
print(f"float add : {tf_add/reps_scalar*1e9:.1f} ns/op")
print(f"Q15 add   : {tq_add/reps_scalar*1e9:.1f} ns/op")
print(f"float mul : {tf_mul/reps_scalar*1e9:.1f} ns/op")
print(f"Q15 mul   : {tq_mul/reps_scalar*1e9:.1f} ns/op")
