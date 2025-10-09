import logging
import textwrap
from typing import Any
from .domain import *
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(threadName)s %(message)s',
    stream=sys.stdout,
)
default_logger = logging.getLogger(__name__)


def format_source_code(source: str) -> str:
    """
    入力されたソースコード文字列から共通の先頭空白を削除し、
    見やすい形に整形して返します。
    """
    dedented = textwrap.dedent(source)
    formatted = "\n" + dedented.strip() + "\n"
    return formatted


def format_value(value: Any) -> str:
    # それ以外の場合、skeleton属性を持つならsource_code()の結果を整形して返す
    if hasattr(value, "best_fn"):
        skeleton_fn = value.best_fn().skeleton
        if callable(skeleton_fn):
            sk = skeleton_fn()
            return format_source_code(str(sk))
    if hasattr(value, "skeleton"):
        skeleton_fn = value.skeleton
        if callable(skeleton_fn):
            sk = skeleton_fn()
            return format_source_code(str(sk))
    # ListまたはTupleの場合は各要素に対して再帰的にformat_valueを適用
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_value(item) for item in value) + "]"
    # return str(value)
    return ""


def default_fn(event: Event) -> None:
    base_message = f"Event: {event.type}"
    if event.type in ["on_best_island_improved"]:
        detail_lines = []

        formatted = format_value(event.payload)
        if "\n" in formatted:
            detail_lines.append(
                f"    Payload:\n{textwrap.indent(formatted, '        ')}")
        else:
            detail_lines.append(f"    Payload: {formatted}")

        complete_message = "\n".join([base_message] + detail_lines)
    else:
        complete_message = base_message
    default_logger.info(complete_message)
    
