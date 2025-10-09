import re
import ast


def fix_wrong_escape(code: str) -> str:
    try:
        code = code.replace("â\x80\x98â\x80\x98â\x80\x98", '"""')
        code = re.sub(r'[^\x00-\x7F]+', ' ', code)
        code = code.encode("utf-8").decode("unicode_escape")
        return code
    except Exception as e:
        raise ValueError("Unicode escape decoding failed.\n", code) from e


def extract_fn_header(code: str) -> str:
    # トリプルクォートのdocstring（シングル・ダブル両方）を削除
    match = re.search(r'def\s+\w+\s*\((.*?)\)', code, re.DOTALL)
    if match:
        # Return the captured group with any extra surrounding whitespace removed.
        return match.group(1).strip()
    raise ValueError("No function header found in the provided code.", code)


def fix_single_quote_line(code: str) -> str:
    """
    単体の '"' が存在する行を検出し、その行の '"' を '\"""' に置換します。

    この関数は、特にdocstringの終了記号として誤って使われた単一の '"' を正しく修正するためのものです。

    Args:
        code (str): 元のコード文字列

    Returns:
        str: 修正後のコード文字列
    """
    # パターンの説明:
    # ^              : 行頭
    # (?P<indent>\s*): インデント（空白文字）をキャプチャ
    # "              : 単体のダブルクオート
    # (?P<tail>\s*)  : 行末までの空白文字をキャプチャ
    # $              : 行末
    pattern = r'^(?P<indent>\s*)"(?P<tail>\s*)$'
    fixed_code = re.sub(pattern, lambda m: m.group(
        'indent') + '"""' + m.group('tail'), code, flags=re.MULTILINE)
    return fixed_code


def remove_docstring(code: str) -> str:
    # トリプルクォートのdocstring（シングル・ダブル両方）を削除
    pattern = r'("""|\'\'\')(.*?)(\1)'
    new_fn_code = re.sub(pattern, '', code, flags=re.DOTALL)
    return new_fn_code


def remove_empty_lines(code: str) -> str:
    # 空行を削除する正規表現
    pattern = r'\n\s*\n'
    new_fn_code = re.sub(pattern, '\n', code)
    return new_fn_code


def set_fn_name(fn_code: str, version: int) -> str:
    pattern = r"^(def\s+)\w+(\s*\(.*?\).*:)"
    new_name = f"equation_v{version}"
    new_fn_code = re.sub(pattern, rf"\1{new_name}\2", fn_code)
    return new_fn_code


def fix_missing_fn_header(code: str, example: str) -> str:
    """
    Fixes the answer if it is missing a function header by extracting the header from the example.

    If the provided answer starts with a 'return' statement (indicating that only the return
    portion of the function is present), this function extracts the function header from the
    example and prepends it to the answer with proper indentation for the function body.
    Otherwise, it returns the answer unchanged.

    Args:
        code (str): The response string that may be missing the function header.
        example (str): A complete function definition used as a reference to extract the header.

    Returns:
        str: The complete function definition with the header, or the original answer if the header is present.

    Raises:
        ValueError: If the answer is missing the header and no valid function header can be found in the example.
    """
    import re

    # Check if the code starts with a "return" statement after stripping leading whitespace.
    if code.lstrip().startswith("return"):
        # Use regex to capture the function header from the example.
        header_match = re.search(
            r'^(def\s+\w+\(.*\).*:)', example, re.MULTILINE)
        if header_match:
            header = header_match.group(0)
            # Indent each line of the code block by 4 spaces to match standard Python formatting.
            indented_code = "\n".join(
                "    " + line if line.strip() != "" else "" for line in code.splitlines())
            # Prepend the header to the indented code block.
            return header + "\n" + indented_code
        else:
            raise ValueError(
                "No function header found in the provided example")
    # If code already includes a function header, return it unchanged.
    return code


def extract_last_function(code: str) -> str:
    """
    Extracts the portion of the code starting from the last function definition header.

    This function searches for the last occurrence of a function definition (a line that starts with "def"
    and ends with a colon). If found, it returns the code starting from that header (thus cutting off anything
    above it). If no function header is found, it returns the original code unchanged so that subsequent
    fixers can handle it.

    Args:
        code (str): The code string to process.

    Returns:
        str: The processed code starting from the last function definition, or the original code if none is found.
    """
    # Regular expression pattern to match a function definition header.
    # It matches lines beginning with 'def ', followed by a function name, parameters in parentheses, and ending with a colon.
    pattern = r'^(def\s+\w+\(.*\).*:)'

    # Find all matches of the function header in the code.
    matches = list(re.finditer(pattern, code, re.MULTILINE))

    # If at least one function header is found, return the substring starting from the last one.
    if matches:
        last_match = matches[-1]
        start_pos = last_match.start()
        return code[start_pos:]
    else:
        # If no function header is found, return the original code to let subsequent fixers process it.
        return code


def fix_missing_header_and_ret(code: str, example: str) -> str:
    """
    Wraps code in a function header extracted from 'example' only if the code is a single assignment
    or a single return statement. If the code already contains a valid function definition, it is
    returned unchanged. This ensures that only cases like:

        'return params[0] * width * wavelength * np.cos(width * np.pi / wavelength) + params[1] * np.sin(width * np.pi / wavelength)'
        'shg_efficieny = params[0] * (params[1] * wavelength - params[2])**2 * width * np.sin(params[3] * width) * np.exp(-params[4] * width) * np.cos(params[5] * wavelength) * np.sin(params[6] * wavelength)'

    are fixed, while complete function definitions remain intact.

    Args:
        code (str): Code that may be missing a function header.
        example (str): A complete function definition used to extract a valid function header.

    Returns:
        str: A complete function definition if applicable, or the original code unchanged.
    """
    # First, try to parse the code to see if it already defines a function.
    try:
        tree = ast.parse(code)
        if tree.body and isinstance(tree.body[0], ast.FunctionDef):
            # Valid function definition found; do not modify.
            return code
    except Exception:
        # If parsing fails here, we assume the code isn't a full function definition.
        pass

    # Attempt to parse the code to check its structure.
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # If parsing fails, return the code unchanged.
        return code

    # Only apply the fix if the code consists of exactly one statement which is either:
    #   - a single assignment statement, or
    #   - a single return statement.
    if (isinstance(tree.body[0], ast.Assign) or isinstance(tree.body[0], ast.Return)):
        # Extract the function header from the example.
        header_match = re.search(
            r'^(def\s+\w+\(.*\).*:)', example, re.MULTILINE)
        if not header_match:
            raise ValueError("提供されたexampleから関数ヘッダーを抽出できませんでした。")
        header = header_match.group(0)

        # Indent the original code by 4 spaces.
        indented_code = "\n".join(
            "    " + line if line.strip() else "" for line in code.splitlines())

        # If it's an assignment statement, add a return statement automatically.
        if isinstance(tree.body[0], ast.Assign):
            assign_node = tree.body[0]
            # Only add return if there's a single target variable.
            if len(assign_node.targets) == 1 and isinstance(assign_node.targets[0], ast.Name):
                var_name = assign_node.targets[0].id
                indented_code += "\n    return " + var_name

        # For a return statement, no additional return is needed.
        return header + "\n" + indented_code

    # If the code doesn't match the two specific cases, return it unchanged.
    return code


def fix_indentation(code: str) -> str:
    """
    与えられたコード内の各行について、先頭の空白数をタブ→スペース変換後に計算し、
    4の倍数になるように四捨五入で調整します。

    例:
      - 5スペース -> 4スペース
      - 3スペース -> 4スペース
      - 7スペース -> 8スペース

    タブは expandtabs(4) を使って4スペース相当に変換するので、大丈夫です。

    Args:
        code (str): 修正前のコード文字列

    Returns:
        str: インデントが調整されたコード文字列
    """
    new_lines = []
    for line in code.splitlines():
        # タブを4スペースに変換
        expanded_line = line.expandtabs(4)
        # 先頭の空白部分を取得
        match = re.match(r'^(\s*)', expanded_line)
        current_indent = match.group(1)  # type: ignore
        num_spaces = len(current_indent)

        # 4の倍数でない場合、四捨五入で最も近い倍数に調整
        if num_spaces % 4 != 0:
            # 四捨五入（半分以上は切り上げ）するために、0.5を足してから整数に変換
            adjusted_spaces = int(num_spaces / 4 + 0.5) * 4
            new_indent = " " * adjusted_spaces
            # 元のインデント部分を新しいインデントに置換
            fixed_line = new_indent + expanded_line[len(current_indent):]
        else:
            fixed_line = expanded_line
        new_lines.append(fixed_line)
    return "\n".join(new_lines)


def force_squash_return_statement(code: str) -> str:
    """
    Finds 'return' statements and forcefully merges subsequent lines
    with equal or greater indentation into a single line, regardless of
    parentheses or backslashes. Skips empty lines and comments during merge.

    Warning: This can significantly reduce readability for valid multi-line returns.
             It assumes the parsing error is related to the multi-line return itself.
    """
    lines = code.splitlines()
    new_code_lines = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped_line = line.lstrip()

        if stripped_line.startswith("return "):
            return_line_indent_str = line[:len(line) - len(stripped_line)]
            start_indent_level = len(return_line_indent_str)
            # Start collecting parts, beginning with the 'return' line content itself
            statement_parts = [stripped_line]
            last_successful_line_index = i
            j = i + 1
            while j < n:
                next_line = lines[j]
                next_stripped = next_line.lstrip()

                # Skip empty lines
                if not next_stripped:
                    j += 1
                    continue
                # Stop merging if we hit a comment line within the potential continuation
                # (Could also choose to skip comments, but stopping is safer)
                if next_stripped.startswith("#"):
                    break

                next_indent_level = len(next_line) - len(next_stripped)

                # Force merge if next line is indented same or more
                if next_indent_level >= start_indent_level:
                    # Append the content (stripped of its own indent)
                    statement_parts.append(next_stripped)
                    last_successful_line_index = j
                    j += 1
                else:
                    # Indent is less, the multi-line statement ends
                    break

            # Combine parts with spaces (remove extra spaces later)
            # Use strip() on each part to avoid joining leading/trailing whitespace
            combined_statement = " ".join(part.strip()
                                          for part in statement_parts)
            # Clean up potential excessive internal spacing
            combined_statement = re.sub(r'\s{2,}', ' ', combined_statement)
            new_code_lines.append(return_line_indent_str + combined_statement)
            i = last_successful_line_index + 1  # Move index past the merged lines
        else:
            # Not a return statement, add line as is
            new_code_lines.append(line)
            i += 1

    return "\n".join(new_code_lines)


def force_squash_assignment_statement(code: str) -> str:
    """
    Finds assignment statements ('=') and forcefully merges subsequent lines
    with STRICTLY GREATER indentation into a single line.
    Skips empty lines and comments during merge. Only triggers if '=' is likely
    an assignment operator.

    This version is safer as it avoids merging subsequent statements at the
    same indent level, focusing on continuations indicated by increased indent.
    """
    lines = code.splitlines()
    new_code_lines = []
    i = 0
    n = len(lines)
    # Regex to find an assignment operator '=', not part of comparison operators
    assignment_regex = re.compile(r'(?<![=<>!])=(?!=)')

    while i < n:
        line = lines[i]
        stripped_line = line.lstrip()

        is_potential_assignment_start = (
            not stripped_line.startswith("#") and
            assignment_regex.search(stripped_line)
        )

        if is_potential_assignment_start:
            assign_line_indent_str = line[:len(line) - len(stripped_line)]
            start_indent_level = len(assign_line_indent_str)

            statement_parts = [stripped_line]
            last_successful_line_index = i
            j = i + 1
            while j < n:
                next_line = lines[j]
                next_stripped = next_line.lstrip()

                # Skip empty lines
                if not next_stripped:
                    j += 1
                    continue
                # Stop merging if we hit a comment line
                if next_stripped.startswith("#"):
                    break

                next_indent_level = len(next_line) - len(next_stripped)

                # --- MODIFIED CONDITION ---
                # Only merge if next line is indented STRICTLY MORE
                if next_indent_level > start_indent_level:
                    statement_parts.append(next_stripped)
                    last_successful_line_index = j
                    j += 1
                else:
                    # Indent is same or less, the potential multi-line statement ends
                    break
            # --- END MODIFICATION ---

            # Combine only if more than one part was collected (i.e., merging happened)
            if len(statement_parts) > 1:
                combined_statement = " ".join(
                    part.strip() for part in statement_parts)
                combined_statement = re.sub(r'\s{2,}', ' ', combined_statement)
                new_code_lines.append(
                    assign_line_indent_str + combined_statement)
                i = last_successful_line_index + 1  # Move index past the merged lines
            else:
                # No lines were merged, just add the original line
                new_code_lines.append(line)
                i += 1
        else:
            # Not a mergeable assignment start, add line as is
            new_code_lines.append(line)
            i += 1

    return "\n".join(new_code_lines)


def get_indent_level(line: str) -> int:
    """
    行の先頭の空白文字の数を返します（インデントレベルの目安）。
    """
    stripped_line = line.lstrip()
    return len(line) - len(stripped_line)


def extract_last_python_block_content(text: str) -> str | None:
    """
    テキストから最後の "```python" と "```" で囲まれたコードブロックの内容を抽出します。
    "```python" マーカーの直後の改行は除去されます。
    """
    # 1. テキスト内で "```python" が出現するすべての開始位置を検索
    block_marker_starts = [match.start()
                           for match in re.finditer(r"```python", text)]

    if not block_marker_starts:
        return None  # "```python" が見つからない

    last_block_marker_start_pos = block_marker_starts[-1]

    # "```python" マーカーの直後から検索を開始
    # len("```python") はマーカー自体の長さ
    content_start_pos = last_block_marker_start_pos + len("```python")

    # content_start_pos 以降で、最初の "```" (終了マーカー) を見つける
    # text[content_start_pos:] は、検索対象を限定するスライス
    end_marker_match = re.search(r"```", text[content_start_pos:])

    if not end_marker_match:
        return None  # 終了の "```" が見つからない

    # コードブロックの内容を抽出
    # end_marker_match.start() はスライスされた部分文字列 text[content_start_pos:] 内での位置
    code_content = text[content_start_pos: content_start_pos +
                        end_marker_match.start()]

    # 一般的に ```python の直後は改行なので、先頭の改行文字がもしあれば1つ除去する
    if code_content.startswith('\n'):
        code_content = code_content[1:]

    return code_content


def find_last_function_in_code(code: str) -> str | None:
    """
    指定されたPythonコード文字列から、最後に定義された関数全体を抽出します。
    関数は "def func_name(...):" で始まり、インデントに基づいて本体が認識されます。
    """
    if not code.strip():  # コードが空か空白のみ
        return None

    lines = code.splitlines()

    candidate_defs_info = []  # 検出された関数定義の情報を格納 (行インデックス、行内容)

    # 3. コードブロック内から関数定義 ("def ...:") の最後のものを探す
    for i, line in enumerate(lines):
        # 行頭の空白も許容し、"def func_name(...):" のパターンにマッチ
        # re.match は行頭からのみ検索。複雑な正規表現は不要。
        if re.match(r"^\s*def.*:$", line):
            candidate_defs_info.append({'line_idx': i})

    if not candidate_defs_info:
        return None  # 関数定義 ("def ...:") が見つからない

    last_def_info = candidate_defs_info[-1]
    last_def_line_idx = last_def_info['line_idx']

    # 4. 最後の関数定義の開始行から、その関数の本体の終わり (return やインデントの終わり) までを見つける
    function_lines = [lines[last_def_line_idx]]  # まずdef行自体を関数の一部として追加
    base_indent_level = get_indent_level(lines[last_def_line_idx])

    for i in range(last_def_line_idx + 1, len(lines)):
        line = lines[i]
        current_indent_level = get_indent_level(line)
        is_empty_line = not line.strip()  # 行が空か空白のみか

        if is_empty_line:
            # 空行の扱い: 基本的には関数に含めるが、インデントが明らかに浅い場合は区切りとみなす。
            # def行よりもインデントが浅い空行は、関数の外の可能性が高い。
            if current_indent_level < base_indent_level:
                # ただし、関数本体が一行も実質的なコードを含んでいない場合（defの直後の浅い空行など）は、
                # この空行を関数の終わりと判断するのは早計かもしれない。
                # しかし、ここではシンプルに「浅いインデントの空行は関数の区切り」とする。
                # 実際には、このような空行は通常、次のブロックとの区切りに使われる。
                # 直前が実質的な行なら区切り
                if len(function_lines) > 1 and function_lines[-1].strip():
                    break
            function_lines.append(line)  # そうでなければ（インデントがbase以上など）、空行も関数の一部
            continue

        # 非空行の場合
        if current_indent_level > base_indent_level:
            function_lines.append(line)  # インデントが深い場合は関数本体の一部
        elif current_indent_level == base_indent_level:
            # インデントがdef行と同じ場合、通常は新しい文の始まり（次の関数定義、クラス定義、トップレベルのコード）。
            # よって、現在の関数はここで終わりとみなす。
            break
        else:  # current_indent_level < base_indent_level
            # インデントが浅くなったら明確に関数の終わり。
            break

    # 関数定義として抽出された行の末尾にある、実質的な内容のない空行（例：インデントのみの行、ただの改行）を削除
    while len(function_lines) > 0 and not function_lines[-1].strip():
        function_lines.pop()

    return "\n".join(function_lines) if function_lines else None

# --- 使用例 ---


def parse_my_text(text_to_parse: str) -> str | None:
    """
    テキストから最後のPythonコードブロック内の最後の関数定義を抽出するメイン処理。
    """
    # ステップ1&2: 最後のPythonコードブロックの内容を抽出
    code_block_content = extract_last_python_block_content(text_to_parse)

    if code_block_content is None:
        # print("エラー: Pythonコードブロックが見つかりませんでした。")
        return None

    # ステップ3&4: コードブロック内から最後の関数定義を抽出
    function_definition = find_last_function_in_code(code_block_content)

    if function_definition is None:
        # print(f"エラー: コードブロック内に関数定義が見つかりませんでした。\nブロック内容:\n{code_block_content}")
        # 関数定義がない場合でも、コードブロックの内容そのものを返したい場合は、
        # ここで `return code_block_content` のように変更できます。
        # 今回は関数定義が見つからない場合は None を返します。
        return None

    return function_definition
