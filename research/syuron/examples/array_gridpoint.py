import jax
import jax.numpy as jnp
from jax import lax
import time
try:
    from syuron import shg2
    shg2.use_gpu()
    print("GPUを有効化しました")
except ImportError:
    print("syuron.shg2モジュールが見つからないため、デフォルト設定を使用します")

# 利用可能なデバイスを確認
print(f"利用可能なJAXデバイス: {jax.devices()}")


def run_computation(s, m, num_steps, use_parallel=True):
    """
    指定されたパラメータで計算を実行し、実行時間を計測する関数

    Parameters:
    s (int): グリッドサイズ
    m (int): 各配列の長さ
    num_steps (int): 漸化式の計算ステップ数
    use_parallel (bool): 並列計算を使用するかどうか

    Returns:
    tuple: (a_results, b_results, execution_time)
    """
    # パラメータ空間の設定
    all_A_arrays = jnp.stack([jnp.ones(m) * (i + 1) for i in range(s)])
    all_B_arrays = jnp.stack([jnp.ones(m) * (i + 4) for i in range(s)])

    # 漸化式のステップ
    def recurrence_step(carry, _):
        a, b = carry
        a_next = a + b
        b_next = a - b
        return (a_next, b_next), (a_next, b_next)

    # グリッドポイント(i,j)での計算
    def compute_at_gridpoint(i, j):
        a_init = all_A_arrays[i]
        b_init = all_B_arrays[j]
        init_state = (a_init, b_init)
        final_state, all_states = lax.scan(
            recurrence_step,
            init_state,
            None,
            length=num_steps
        )
        a_final, b_final = final_state
        return a_final, b_final

    if use_parallel:
        # 並列計算（vmap）
        compute_row = jax.vmap(compute_at_gridpoint, in_axes=(None, 0))
        compute_grid = jax.vmap(compute_row, in_axes=(0, None))

        # コンパイル（初回実行の時間を除外）
        i_indices = jnp.arange(s)
        j_indices = jnp.arange(s)
        _ = compute_grid(i_indices, j_indices)

        # 実行時間の測定
        start_time = time.time()
        a_results, b_results = compute_grid(i_indices, j_indices)
        # 計算完了を確実にする（JAXの遅延評価のため）
        jax.block_until_ready(a_results)
        end_time = time.time()
    else:
        # 逐次計算（for loop）
        a_results = jnp.zeros((s, s, m))
        b_results = jnp.zeros((s, s, m))

        # コンパイル
        _ = compute_at_gridpoint(0, 0)

        # 実行時間の測定
        start_time = time.time()
        for i in range(s):
            for j in range(s):
                a_result, b_result = compute_at_gridpoint(i, j)
                a_results = a_results.at[i, j].set(a_result)
                b_results = b_results.at[i, j].set(b_result)
        jax.block_until_ready(a_results)
        end_time = time.time()

    execution_time = end_time - start_time
    return a_results, b_results, execution_time


def compare_performance():
    """異なるパラメータでの計算時間を比較する関数"""
    s_values = [1, 10, 50, 100, 500]  # グリッドサイズ
    m_values = [1, 10, 100]  # 配列の長さ
    steps_values = [1, 10, 100]  # 計算ステップ数

    print("\n1. sを変えた場合の計算時間（m=10, steps=10）:")
    print(f"{'s':>6} | {'並列時間(秒)':>15} | {'逐次時間(秒)':>15} | {'速度向上率':>10}")
    print("-" * 60)
    for s in s_values:
        _, _, parallel_time = run_computation(s, 10, 10, True)
        # 小さなsの場合のみ逐次計算を行う
        if s <= 100:
            _, _, sequential_time = run_computation(s, 10, 10, False)
            speedup = sequential_time / parallel_time
            print(
                f"{s:6d} | {parallel_time:15.6f} | {sequential_time:15.6f} | {speedup:10.2f}x")
        else:
            print(f"{s:6d} | {parallel_time:15.6f} | {'N/A':>15} | {'N/A':>10}")

    print("\n2. mを変えた場合の計算時間（s=10, steps=10）:")
    print(f"{'m':>6} | {'並列時間(秒)':>15} | {'逐次時間(秒)':>15} | {'速度向上率':>10}")
    print("-" * 60)
    for m in m_values:
        _, _, parallel_time = run_computation(10, m, 10, True)
        _, _, sequential_time = run_computation(10, m, 10, False)
        speedup = sequential_time / parallel_time
        print(
            f"{m:6d} | {parallel_time:15.6f} | {sequential_time:15.6f} | {speedup:10.2f}x")

    print("\n3. 計算ステップ数を変えた場合の計算時間（s=10, m=10）:")
    print(f"{'steps':>6} | {'並列時間(秒)':>15} | {'逐次時間(秒)':>15} | {'速度向上率':>10}")
    print("-" * 60)
    for steps in steps_values:
        _, _, parallel_time = run_computation(10, 10, steps, True)
        _, _, sequential_time = run_computation(10, 10, steps, False)
        speedup = sequential_time / parallel_time
        print(
            f"{steps:6d} | {parallel_time:15.6f} | {sequential_time:15.6f} | {speedup:10.2f}x")


# メイン実行
if __name__ == "__main__":
    # 簡単なテスト実行
    print("簡単なテスト実行（s=2, m=3, steps=5）:")
    a_results, b_results, exec_time = run_computation(2, 3, 5)
    print(f"実行時間: {exec_time:.6f}秒")
    print(f"結果の形状: a_results.shape = {a_results.shape}")
    print(f"a_results[0,0] = {a_results[0,0]}")
    print(f"a_results[1,1] = {a_results[1,1]}")

    # 様々なパラメータでのパフォーマンス比較
    compare_performance()
