import math

def int_to_bit_array(num, n_bits):
    """整数をnビットのリストに変換します（例: 5, 3ビット -> [1,0,1]）。"""
    return [int(bit) for bit in bin(num)[2:].zfill(n_bits)]

def bit_array_to_str(bit_array):
    """ビットのリストを文字列に変換します（例: [1,0,1] -> "101"）。"""
    return "".join(map(str, bit_array))

def xor_bit_arrays(arr1, arr2):
    """2つのビットリストのビット単位XORを計算します。"""
    if len(arr1) != len(arr2):
        raise ValueError("ビット配列の長さが異なります。")
    return [a ^ b for a, b in zip(arr1, arr2)]

def dot_product_mod2(arr1, arr2):
    """2つのビットリストの内積を計算し、結果をmodulo 2で返します。"""
    if len(arr1) != len(arr2):
        raise ValueError("ビット配列の長さが異なります。")
    prod_sum = sum(a * b for a, b in zip(arr1, arr2))
    return prod_sum % 2

def verify_identity(n_bits):
    """
    指定されたビット数 n_bits で等式を検証します。
    左辺: (1/2^n) * sum_y (sum_x (-1)^((s XOR y) . x)) |y>
    右辺: |s>
    """
    num_states = 2**n_bits
    print(f"検証開始: n = {n_bits} ビット (状態数: {num_states})")
    print("-" * 40)

    all_s_verified = True

    # すべての可能なターゲット状態 |s> についてループ
    for s_val in range(num_states):
        s_target_bits = int_to_bit_array(s_val, n_bits)
        s_target_str = bit_array_to_str(s_target_bits)
        print(f"\nターゲット状態 |s> = |{s_target_str}> (s = {s_val}) の検証中...")

        # 左辺によって計算される状態ベクトル (各 |y> の係数)
        # result_coeffs[y_val] は |y> の係数
        result_coeffs = [0.0] * num_states

        # |y> の係数を計算するために、すべての y についてループ
        for y_val in range(num_states):
            y_bits = int_to_bit_array(y_val, n_bits)
            # y_str = bit_array_to_str(y_bits) # デバッグ用

            # k = s_target XOR y
            k_bits = xor_bit_arrays(s_target_bits, y_bits)

            # 内側の和: sum_x (-1)^(k . x)
            inner_sum_val = 0
            # すべての x についてループ
            for x_val in range(num_states):
                x_bits = int_to_bit_array(x_val, n_bits)
                
                dot_prod = dot_product_mod2(k_bits, x_bits)
                term = (-1)**dot_prod
                inner_sum_val += term
            
            # y の係数 = (1/2^n) * inner_sum_val
            coeff_y = (1.0 / num_states) * inner_sum_val
            result_coeffs[y_val] = coeff_y
            
            # print(f"  |y> = |{y_str}> (y={y_val}): k={bit_array_to_str(k_bits)}, inner_sum={inner_sum_val}, coeff={coeff_y}") # 詳細デバッグ

        # 検証: result_coeffs が |s_target_bits> を表しているか
        # つまり、y_val == s_val のとき係数が1で、それ以外は0か？
        current_s_verified = True
        print(f"  計算された係数 (ターゲット |{s_target_str}>):")
        expected_coeffs = [0.0] * num_states
        expected_coeffs[s_val] = 1.0
        
        for i in range(num_states):
            y_state_str = bit_array_to_str(int_to_bit_array(i, n_bits))
            is_correct = math.isclose(result_coeffs[i], expected_coeffs[i], abs_tol=1e-9)
            print(f"    coeff for |{y_state_str}>: {result_coeffs[i]:.4f} (期待値: {expected_coeffs[i]:.1f}) - {'OK' if is_correct else 'NG'}")
            if not is_correct:
                current_s_verified = False
                all_s_verified = False
        
        if current_s_verified:
            print(f"  結果: |s> = |{s_target_str}> は正しく構成されました。")
        else:
            print(f"  結果: |s> = |{s_target_str}> の構成に誤りがあります。")

    print("-" * 40)
    if all_s_verified:
        print(f"\nすべてのターゲット |s> について、n = {n_bits} で等式が検証されました！")
    else:
        print(f"\nエラー: n = {n_bits} で等式が検証されませんでした。")

# --- プログラムの実行 ---
# n_bits の値を小さく設定してください (例: 1, 2, 3)。
# n_bits が大きくなると計算量が急激に増大します (2^(3*n) のオーダー)。
N_BITS_TO_TEST = 4
verify_identity(N_BITS_TO_TEST)

print("\n別の例: n=1")
verify_identity(1)

# print("\n注意: n=3 以上は計算に時間がかかります")
# verify_identity(3) # 例: n=3 (状態数8、ループ回数 8*8*8 = 512 per s)