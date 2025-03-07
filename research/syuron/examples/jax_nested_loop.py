import time
import jax
from jax import lax

jax.config.update('jax_platform_name', 'gpu')  # または 'gpu'
print(f"使用デバイス: {jax.devices()}")

# dx/dt = 0.1*x の前進オイラー法
dt = 0.00001


def inner_step(x, _):
    return x + dt * (0.1 * x), None


def outer_step(x, _):
    x, _ = lax.scan(inner_step, x, None, length=1000)
    return x, None


def run_simulation(x0, n_outer_steps):
    x_final, _ = lax.scan(outer_step, x0, None, length=n_outer_steps)
    return x_final


x0 = 1.0

start = time.time()
result = run_simulation(x0, 1000)
end = time.time()

print(f"計算結果: {result}")
print(f"実行時間: {end - start:.6f}秒")
