import json


class TreeStyleJSONEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_indent = 0
        self.indent_str = " "  # 2スペースインデント

    def encode(self, o):
        if isinstance(o, (dict, list)):
            self.current_indent = 0
            return self._encode(o)
        return json.JSONEncoder.encode(self, o)

    def _encode(self, obj):
        if isinstance(obj, dict):
            if not obj:
                return "{}"

            # ディレクトリ/ファイル構造の特別処理
            if set(obj.keys()) <= {"type", "name", "contents"}:
                parts = [f'"type":"{obj["type"]}"', f'"name":"{obj["name"]}"']

                if "contents" in obj:
                    self.current_indent += 1
                    contents_str = self._encode(obj["contents"])
                    self.current_indent -= 1
                    parts.append(f'"contents":{contents_str}')

                return "{" + ",".join(parts) + "}"

            # 一般的な辞書の処理
            self.current_indent += 1
            parts = []
            for key, value in obj.items():
                parts.append(
                    f'{self.indent_str * self.current_indent}"{key}":{self._encode(value)}')
            self.current_indent -= 1
            return "{\n" + ",\n".join(parts) + "\n" + self.indent_str * self.current_indent + "}"

        elif isinstance(obj, list):
            if not obj:
                return "[]"

            self.current_indent += 1
            parts = []
            for item in obj:
                encoded_item = self._encode(item)
                parts.append(self.indent_str *
                             self.current_indent + encoded_item)
            self.current_indent -= 1

            return "[\n" + ",\n".join(parts) + "\n" + self.indent_str * self.current_indent + "]"

        elif isinstance(obj, str):
            return f'"{obj}"'
        else:
            return json.JSONEncoder.encode(self, obj)
