from functools import partial
import time
import numpy as np
from scipy.signal import czt as scipy_czt
import jax.numpy as jnp
import jax
from jax import config
# 信号処理する上でこれは必須
config.update("jax_enable_x64", True)


# --- JAXのカスタムCZT関数 ---
# 'm'と'fft_len'を静的引数として指定
@partial(jax.jit, static_argnames=('m', 'fft_len'))
def custom_czt_jax(x, m, fft_len, w=None, a=1.0):
    """
    Bluesteinのアルゴリズムに基づいた、JAXによるCZT実装
    """
    n = x.shape[-1]
    if w is None:
        w = jnp.exp(-2j * jnp.pi / m)

    # fft_lenは静的引数として外から渡されるため、ここでの計算は不要
    n_range = jnp.arange(n)
    y = x * (a**-n_range) * w**(n_range**2 / 2)

    k_range_full = jnp.arange(-(n - 1), m)
    h = w**(-(k_range_full**2) / 2)

    # fft_lenが静的な値なので、このFFTはコンパイル可能になる
    Y = jnp.fft.fft(y, n=fft_len)
    H = jnp.fft.fft(h, n=fft_len)
    conv_result = jnp.fft.ifft(Y * H)

    k_range_out = jnp.arange(m)
    final_chirp = w**(k_range_out**2 / 2)

    # `conv_result`の範囲を正しくスライス
    return conv_result[n-1:n-1+m] * final_chirp


# --- パフォーマンス比較 ---
# パラメータ設定
N_signal = 50000
M_output = 100
w_param = np.exp(-1j * np.pi / M_output / 2)
a_param = 1.0

# データ生成
x_np = np.random.randn(N_signal).astype(np.complex64)
x_jax = jnp.array(x_np)

# ★★★ 修正点 ★★★
# FFTサイズを関数の外側で、Pythonの整数として計算する
required_len = N_signal + M_output - 1
# 次の2のべき乗を計算する効率的な方法
fft_len_static = 1 << (required_len - 1).bit_length()


# --- 1. JAX版の実行 ---
print("🚀 JAX版の実行時間を計測...")
print("   (初回実行でJITコンパイル中...)")
# JITコンパイル
_ = custom_czt_jax(x_jax, m=M_output, fft_len=fft_len_static,
                   w=w_param, a=a_param).block_until_ready()
print("   (コンパイル完了)")

start_time = time.perf_counter()
jax_result = custom_czt_jax(
    x_jax, m=M_output, fft_len=fft_len_static, w=w_param, a=a_param).block_until_ready()
end_time = time.perf_counter()
jax_duration = (end_time - start_time) * 1000
print(f"   JAX 実行時間: {jax_duration:.4f} ms")


print("-" * 30)


# --- 2. SciPy版の実行 ---
print("🔬 SciPy版の実行時間を計測...")
start_time = time.perf_counter()
scipy_result = scipy_czt(x_np, m=M_output, w=w_param, a=a_param)
end_time = time.perf_counter()
scipy_duration = (end_time - start_time) * 1000
print(f"   SciPy 実行時間: {scipy_duration:.4f} ms")

print("-" * 30)

# --- 速度の比較 ---
if jax_duration > 0 and scipy_duration > 0:
    if jax_duration < scipy_duration:
        speed_ratio = scipy_duration / jax_duration
        print(f"✅ JAXはSciPyの約 {speed_ratio:.2f} 倍 高速でした。")
    else:
        speed_ratio = jax_duration / scipy_duration
        print(f"🐌 SciPyはJAXの約 {speed_ratio:.2f} 倍 高速でした。")

print("-" * 30)

# --- 3. 結果の数値的な比較 ---
print("🔍 結果の数値的な比較...")

# SciPyのNumPy配列をJAX配列に変換
scipy_result_jax = jnp.asarray(scipy_result)

# jnp.allclose() を使って、浮動小数点数の誤差を許容しつつ比較
# atol (absolute tolerance) と rtol (relative tolerance) を調整して比較精度を制御
are_close = jnp.allclose(jax_result, scipy_result_jax, atol=1e-5, rtol=1e-5)

# 結果を出力
if are_close:
    print("👍 JAXとSciPyの結果は数値的にほぼ一致しました。")
else:
    print("🤔 JAXとSciPyの結果が異なります。")
    # デバッグ用に差の最大値を出力
    max_abs_diff = jnp.max(jnp.abs(jax_result - scipy_result_jax))
    print(f"   最大絶対誤差: {max_abs_diff}")
