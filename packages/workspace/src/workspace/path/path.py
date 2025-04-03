from workspace import config
import os

WORKSPACE_ROOT = config.WORKSPACE_ROOT


class Path(str):
    _abs: str
    _display: str

    def __new__(cls, value: str, *, _abs: str | None = None, _display: str | None = None) -> 'Path':
        # 内部用引数 _abs/_display が渡されていればそれを使う
        if _abs is not None:
            abs_path = os.path.normpath(_abs)
            # _display が指定されていなければ WORKSPACE_ROOT 基準で計算
            display = os.path.normpath(
                _display) if _display is not None else os.path.relpath(abs_path, WORKSPACE_ROOT)
        else:
            # ユーザからの呼び出し
            if os.path.isabs(value):
                if not value.startswith(WORKSPACE_ROOT):
                    raise ValueError("パスが workspace の外にあります")
                abs_path = os.path.normpath(value)
                display = os.path.relpath(abs_path, WORKSPACE_ROOT)
            else:
                display = os.path.normpath(value)
                abs_path = os.path.join(WORKSPACE_ROOT, display)
        obj = super().__new__(cls, display)
        obj._abs = abs_path      # 絶対パス
        obj._display = display   # 表示用パス（初期は WORKSPACE_ROOT 基準）
        return obj

    def abs(self) -> str:
        return self._abs

    def rel2(self, base: 'Path') -> 'Path':
        """
        新しい基点 base からの相対パスを表示用パスとして返す
        """
        new_display = os.path.relpath(self._abs, base.abs())
        return Path(new_display, _abs=self._abs, _display=new_display)

    def dir(self) -> 'Path':
        """
        自身のディレクトリを返す（表示用パスも同様に親ディレクトリに変更）
        """
        dir_abs = os.path.dirname(self._abs)
        new_display = os.path.dirname(self._display)
        return Path(new_display, _abs=dir_abs, _display=new_display)

    def __truediv__(self, other: str) -> 'Path':
        """
        パス結合。絶対パスは単純に連結し、表示用パスも同様に連結する
        """
        new_abs = os.path.join(self._abs, other)
        new_display = os.path.join(self._display, other)
        return Path(new_display, _abs=new_abs, _display=new_display)
