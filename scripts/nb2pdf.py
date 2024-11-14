#!/usr/bin/env python3

import argparse
from nbconvert import HTMLExporter
from playwright.sync_api import sync_playwright
from traitlets.config import Config
from nbconvert import HTMLExporter
import time


def convert_notebook_to_html(notebook_path: str) -> str:
    c = Config()
    c.HTMLExporter.exclude_input_prompt = True
    c.HTMLExporter.exclude_output_prompt = True
    html_exporter = HTMLExporter(config=c)
    body, _ = html_exporter.from_file(notebook_path)
    return body


def save_html_as_pdf(html: str, output_path: str):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html)
        page.add_style_tag(
            content="* { font-family: 'Noto Sans CJK JP', 'Hiragino Kaku Gothic ProN', 'メイリオ', sans-serif !important; }")
        page.wait_for_load_state('networkidle')
        time.sleep(0.5)  # wait for mathjax
        page.pdf(
            path=output_path,
            scale=0.9,
            format="A4"
        )
        browser.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Jupyter notebook to PDF')
    parser.add_argument('-f', '--file', help='Notebook file path')
    parser.add_argument(
        '-o', '--output', help='Output PDF file path (default: output.pdf)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    html = convert_notebook_to_html(args.file)
    if args.output is None:
        args.output = args.file.replace(".ipynb", ".pdf")
    save_html_as_pdf(html, args.output)
    print("PDFを保存しました:", args.output)
