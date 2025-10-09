from funsearch import function
from funsearch import profiler
from funsearch import llmsr
import time
import ast


def test_parse():
    # これ以外の edge ケース例で skeleton の出力が複素数の時とかあるけど、それは普通に式が間違ってるから無視
    edge_cases = [
        # gemma3:12b が生成したバグコード
        "def equation_v1(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:\n    ''' Mathematical function for acceleration in a damped nonlinear oscillator\n\n    Args:\n        x: A numpy array representing observations of current position.\n        v: A numpy array representing observations of velocity.\n        params: Array of numeric constants or parameters to be optimized\n\n    Return:\n        A numpy array representing acceleration as the result of applying the mathematical function to the inputs.\n    '''\n    k = params[0]  # Damping coefficient\n    c = params[1]  # Spring constant (if applicable)\n    F_t = params[2]  # Driving force, assumed constant for simplicity\n\n    dv = -k * x - c * v + F_t\n    return dv\n",  # normal case
        "def equation_v1(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:\n    ''' Mathematical function for acceleration in a damped nonlinear oscillator\n\n    Args:\n        x: A numpy array representing observations of current position.\n        v: A numpy array representing observations of velocity.\n        params: Array of numeric constants or parameters to be optimized\n\n    Return:\n        A numpy array representing acceleration as the result of applying the mathematical function to the inputs.\n    '''\n    k = params[0]  # Damping coefficient\n    c = params[1]  # Spring constant (if applicable)\n    F_t = params[2]  # Driving force, assumed constant for simplicity\n\n    dv = -k * x - c * v + F_t\n    return dv\n```",
        "def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:\n    E = params[0]  # Young's modulus\n    CTE = params[1]  # Coefficient of thermal expansion\n    stress = E * strain + CTE * temp\n    return stress",
        'def equation_v1(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:\n   """\n   Mathematical function for stress in Aluminium rod\n\n   Args:\n       strain: A numpy array representing observations of strain.\n       temp: A numpy array representing observations of temperature.\n       params: Array of numeric constants or parameters to be optimized\n\n   Return:\n       A numpy array representing stress as the result of applying the mathematical function to the inputs.\n   "\n   stress = params[0] * x + params[1] * v\n   return stress',
        'return params[0] * width * wavelength * np.sin(width/wavelength) + params[1] * width * wavelength',  # return の中だけ書いて来る時結構あるから対応できるようにする
        'return params[0] * width * wavelength * np.cos(width * np.pi / wavelength) + params[1] * np.sin(width * np.pi / wavelength)',
        'shg_efficieny = params[0] * (params[1] * wavelength - params[2])**2 * width * np.sin(params[3] * width) * np.exp(-params[4] * width) * np.cos(params[5] * wavelength) * np.sin(params[6] * wavelength)',
        'def equation_v3(width: np.ndarray, wavelength: np.ndarray, params: np.ndarray) -> np.ndarray:\n   """ \n   Mathematical function for shg efficiency\n\n   Args:\n       width: A numpy array representing periodic domain width\n       wavelength: A numpy array representing wavelength.\n       params: Array of numeric constants or parameters to be optimized\n\n   Return:\n       A numpy array representing shg efficiency as the result of applying the mathematical function to the inputs.\n   """\n    # Physical insight: SHG efficiency is often a complex function of both width and wavelength.\n    # It can involve quadratic and higher-order terms to account for dispersion and interaction strength.\n    # Also, the interaction strength may vary with wavelength, often modeled by a Gaussian function.\n    # Interaction strength modeled by a Gaussian function centered at a specific wavelength.\n    interaction_strength = params[0] * np.exp(-params[1] * (wavelength - params[2])**2)\n    # SHG efficiency is a combination of width-dependent and wavelength-dependent terms,\n    # modulated by the wavelength-dependent interaction strength.\n    shg_efficiency = interaction_strength * (params[3] * width**2 + params[4] * wavelength**2 + params[5] * width * wavelength)\n    return shg_efficiency',  # indent多少ミスってるやつ
        'def equation_v3(width: np.ndarray, wavelength: np.ndarray, params: np.ndarray) -> np.ndarray:\n   """ \n   Mathematical function for shg efficiency\n\n   Args:\n       width: A numpy array representing periodic domain width\n       wavelength: A numpy array representing wavelength.\n       params: Array of numeric constants or parameters to be optimized\n\n   Return:\n       A numpy array representing shg efficiency as the result of applying the mathematical function to the inputs.\n   """\n    # Physical considerations: SHG efficiency often depends on the square of a parameter related to the refractive index difference.\n    # This model introduces a term that considers the square of a wavelength-dependent factor.\n    return params[0] * width * (1 + params[1] * wavelength**2) + params[2] * wavelength**2\n',
        # qwen2.5-coder:7b が生成したバグコード
        'def equation_v1(width: np.ndarray, wavelength: np.ndarray, params: np.ndarray) -> np.ndarray:\\n    # Assuming SHG efficiency is a function of both width and wavelength\\n    return params[0] * width ** 2 + params[1] * wavelength ** 2',
        'def equation_v2(width: np.ndarray, wavelength: np.ndarray, params: np.ndarray) -> np.ndarray:\\n    \\""\\"\n    Mathematical function for shg efficiency\n    \\n    Args:\n        width: A numpy array representing periodic domain width\n        wavelength: A numpy array representing wavelength.\n        params: Array of numeric constants or parameters to be optimized\n    \\n    Return:\n        A numpy array representing shg efficiency as the result of applying the mathematical function to the inputs.\n    \\""\\"\n    return params[0] * width**3 + params[1] * wavelength**2 + params[2] * np.sin(width) + params[3] * np.cos(wavelength)',
        'def equation_v2(width: np.ndarray, wavelength: np.ndarray, params: np.ndarray) -> np.ndarray:\\n    return params[0] * width + params[1] * wavelength\\n',
        # 'equation_v2(width: np.ndarray, wavelength: np.ndarray, params: np.ndarray) -> np.ndarray:\n    return params[0] * width**3 + params[1] * wavelength**2 + params[2] * width * wavelength',
        'def equation_v2(width: np.ndarray, wavelength: np.ndarray, params: np.ndarray) -> np.ndarray:\\n    """\n    Improved mathematical function for SHG efficiency\n    \\n    Args:\\n        width: A numpy array representing periodic domain width\\n        wavelength: A numpy array representing wavelength.\\n        params: Array of numeric constants or parameters to be optimized\\n    \\n    Return:\\n        A numpy array representing SHG efficiency as the result of applying the mathematical function to the inputs.\\n    """\\n    shg_efficiency = params[0] * width**3 + params[1] * wavelength**2 + params[2] * np.sin(wavelength) + params[3] * width * wavelength**3\\n    return shg_efficiency',
        'def equation_v2(width: np.ndarray, wavelength: np.ndarray, params: np.ndarray) -> np.ndarray:\\n\\n    alpha = params[0]\\n    beta = params[1]\\n    gamma = params[2]\\n    delta = params[3]\\n    epsilon = params[4]\\n    zeta = params[5]\\n    eta = params[6]\\n    theta = params[7]\\n\\n    efficiency = alpha * width**3 + beta * wavelength**2 + gamma * width * wavelength + delta * np.sin(width) + epsilon * np.cos(wavelength) + zeta * width * wavelength ** 0.5 + eta * np.exp(-theta * (width - wavelength))\\n\\n    return efficiency',
        "def equation_v2(width: np.ndarray, wavelength: np.ndarray, params: np.ndarray) -> np.ndarray:\n    # Assuming SHG efficiency is a function of both width and wavelength\n    # This model introduces a term that considers the square of a wavelength-dependent factor.\n    return params[0] * width**3 + params[1] * wavelength**2 + params[2] * width * wavelength\n",
        # qwen2.5-coder:14b が生成したバグコード
        'def equation_v1(width: np.ndarray, wavelength: np.ndarray, params: np.ndarray) -> np.ndarray:\\n    return params[0] * width**params[1] * wavelength**params[2]',
        "def equation_v1(width: np.ndarray, wavelength: np.ndarray, params: np.ndarray) -> np.ndarray:\n    # Assuming SHG efficiency depends on a more complex relationship than a simple linear combination.\n    # For instance, let's consider a model where SHG efficiency is influenced by the ratio of width to wavelength,\n    # and also includes some non-linear parameters.\n    \n    # Example: Using a Gaussian function for simplicity, which could be adjusted based on physical insights or experimental data.\n    width_normalized = width / np.max(width)\n    wavelength_normalized = wavelength / np.max(wavelength)\n    \n    efficiency = params[0] * np.exp(-params[1] * (width_normalized - params[2])**2) *\n                    np.exp(-params[3] * (wavelength_normalized - params[4])**2) +\n                    params[5] * np.sin(params[6] * width_normalized + params[7]) +\n                    params[8] * np.cos(params[9] * wavelength_normalized)\n    \n    return efficiency",
        "def equation_v2(width: np.ndarray, wavelength: np.ndarray, params: np.ndarray) -> np.ndarray:\n    â\x80\x98â\x80\x98â\x80\x98 \n    Mathematical function for shg efficiency\n\n    Args:\n        width: A numpy array representing periodic domain width\n        wavelength: A numpy array representing wavelength.\n        params: Array of numeric constants or parameters to be optimized\n\n    Return:\n        A numpy array representing shg efficiency as the result of applying the mathematical function to the inputs.\n    â\x80\x98â\x80\x98â\x80\x98\n    # Incorporate higher-order terms and physical constraints for better accuracy\n    shg_efficiency = params[0] * width**2 + params[1] * wavelength**-1 + params[2] * width * wavelength + params[3] * np.sin(params[4] * width) + params[5] * np.cos(params[6] * wavelength)\n\n    # Apply a physical constraint that efficiency cannot exceed 1\n    shg_efficiency = np.clip(shg_efficiency, 0, 1)\n\n    return shg_efficiency",
        "def equation_v2(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:\n    # Calculate the bacterial growth rate considering non-linear effects\n    return (params[0] * b + params[1] * s + \n            params[2] * (temp - 25)**2 + params[3] * (pH - 7.0)**2 + \n            params[4] * np.exp(-params[5] * b) + params[6] * np.exp(-params[7] * s)) * \n            (1 + params[8] * (b / (params[9] + b)))",
        # "def equation_v2(width: np.ndarray, wavelength: np.ndarray, params: np.ndarray) -> np.ndarray:\n    # Phase shift with group velocity dispersion and higher order terms\n    phase_shift = np.sin(2 * np.pi * (width + params[0]) / (params[1] * wavelength**2))\n    phase_shift += params[2] * np.cos(params[3] * width / wavelength)\n    # Gaussian envelope adjusted for nonlinear effects in domain width and wavelength\n    gaussian_envelope = np.exp(-(width - params[4])**2 / (2 * (params[5] + params[6] * wavelength**2))\n    # Sinc function enhanced to account for phase matching and periodicity more accurately\n    sinc_argument = (phase_shift - params[7]) / (params[8] + params[9] * np.sin(params[10] * width) / wavelength)\n    adjusted_sinc = np.sinc(sinc_argument) * gaussian_envelope\n    # Nonlinear correction term to account for frequency dispersion and phase modulation effects\n    correction_term = 1 + params[11] * ((np.cos(params[12] * width + params[13]) / wavelength)**2)\n    correction_term += params[15] * ((np.sin(params[14] * width + params[16]) / wavelength**3)**2) + params[18] * (width - params[19])**2 / wavelength**4\n    correction_term += params[20] * np.tanh(params[21] * width) + params[22] * ((wavelength - params[23])**2 / wavelength**2)\n    # Final efficiency calculation with a refined scale factor for accuracy and proportionality\n    efficiency = params[24] * (adjusted_sinc ** 2) * correction_term\n    return efficiency", # かっこが正しく閉じられていないのは想定解が確定しないから詰んでる
        # phi4 が生成したバグコード
        "def equation_v1(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:\n    ''' \n    Mathematical function for bacterial growth rate\n    Args:\n        b: A numpy array representing observations of population density of the bacterial species.\n        s: A numpy array representing observations of substrate concentration.\n        temp: A numpy array representing observations of temperature.\n        pH: A numpy array representing observations of pH level.\n        params: Array of numeric constants or parameters to be optimized\n    Return:\n        A numpy array representing bacterial growth rate as the result of applying the mathematical function to the inputs.\n    '''\n    return params[0] * b / (params[1] + b) * s / (params[2] + s) * \n           np.exp(params[3] * (temp - 37)) * \n           np.exp(-((pH - 7) ** 2) / (2 * params[4]**2))",  # (return の後の長いコードに改行が入っている)
        "def add_one_plus_one():\n\treturn 1 + 1",
    ]
    engine = llmsr.PyMutationEngine("", "")
    for demo_fn in edge_cases:
        try:
            parsed = engine._parse_answer(demo_fn, edge_cases[0])
            ast.parse(parsed)
        except Exception as e:
            print(f"Parsed failed: \n{demo_fn}")
            print(f"Error: {e}")
            return

    print(f"All equations parsed successfully")


# if __name__ == "__main__":
#     test_parse()
    
# テスト用の文字列 (ユーザーが提示したエラーケースを含む)
test_text_with_error_case = """
other text
```python
def func_in_block_1():
    return "first"
```
some more text
```python
  phi = volume_fraction
  E_composite = E_m * (1 - phi)**params[0] + E_f * phi**params[1] + params[2] * E_m * E_f * phi * (1 - phi)
  return E_composite
```
"""

test_text_with_function = """
many text

# last code block
```python
Many CODE

def earlier_func_in_last_block():
    pass # something

def last_defined_function(args):
    # This is the target function
    if True:
        return True # with a comment
    # Another line
    return False
```

other text
"""

demo = '''
'```python\ndef equation_v1(volume_fraction: np.ndarray, params: np.ndarray, E_m=4.84, E_f=117.64) -> np.ndarray:\n    """ \n    Mathematical function for the tensile modulus of a particle-filler rubber composite.\n\n    This function aims to model the relationship between the filler volume fraction (phi)\n    and the experimentally observed tensile modulus of the composite material.\n    The core of this function, to be evolved by FunSearch, **must explicitly use**\n    the provided tensile modulus of the matrix (E_m = 4.84) and the filler (E_f = 117.64)\n    as variables within the mathematical expression. These constants are passed as arguments\n    to this function and are expected to be directly part of the formula.\n\n    Args:\n        volume_fraction: A numpy array representing the filler volume fraction (phi).\n        params: Array of numeric constants or parameters (at most MAX_NPARAMS)\n                to be optimized by the fitting process. These parameters will be\n                used within the function skeleton.\n        E_m: Tensile modulus of the matrix (fixed at 4.84). This value **must be used**\n             in the generated equation.\n        E_f: Tensile modulus of the filler (fixed at 117.64). This value **must be used**\n             in the generated equation.\n\n    Return:\n        A numpy array representing the predicted tensile modulus of the composite (E_composite).\n        The returned value is the result of applying the mathematical function,\n        which incorporates phi, params, E_m, and E_f.\n    """\n    phi = volume_fraction\n    E_composite = E_m * (1 - phi)**params[0] * (E_f/E_m)**(phi * params[1])\n    return E_composite\n```'
'''

# 1. 関数定義がないコードブロックのテスト
print("--- Test 1 (No function definition in last block) ---")
result1 = llmsr.parse_my_text(demo)
if result1:
    print(result1)
else:
    # このケースでは関数定義が見つからないため、Noneが返るのが期待される動作
    print("Expected: No function found, Result: None (Correct for this implementation)")
    # 元のコードで例外を発生させていた箇所に相当。
    # この関数の呼び出し側で、Noneが返ってきた場合の処理を記述します。
    # 例えば:
    # if result1 is None:
    #   raise Exception("Gemini response parse error (no function found)", 
    #                   f"Input block was: {extract_last_python_block_content(test_text_with_error_case)}")


print("\n--- Test 2 (With function definition) ---")
# 2. 関数定義があるコードブロックのテスト
result2 = llmsr.parse_my_text(test_text_with_function)
if result2:
    print(result2)
else:
    print("Function not found (unexpected for this test case).")

