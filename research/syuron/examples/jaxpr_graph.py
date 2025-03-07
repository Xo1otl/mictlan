import jax
import jax.numpy as jnp

def f(x, y):
    return x + y  # 実際の重い計算だと仮定

def g(val, z):
    return val * z

@jax.jit
def compute(x, y, z):
    return g(f(x, y), z)

# ダミーの入力でjaxprを確認
x, y, z = jnp.array(1.), jnp.array(2.), jnp.array(3.)
print(jax.make_jaxpr(compute)(x, y, z))
