from playwright.sync_api import sync_playwright
import os
from datetime import datetime


def save_webpage_as_pdf(url: str, output_path: str = ""):
    """
    指定したWebページをPDFとして保存する

    Args:
        url (str): PDFとして保存したいWebページのURL
        output_path (str, optional): 出力PDFのパス。Noneの場合は現在時刻でファイル名を生成
    """

    if output_path is None:
        # 出力パスが指定されていない場合、タイムスタンプでファイル名を生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"webpage_{timestamp}.pdf"

    with sync_playwright() as p:
        # ブラウザの起動（フォント関連の設定を含む）
        browser = p.chromium.launch(
            args=[
                '--font-render-hinting=medium',
                '--enable-font-antialiasing',
            ]
        )

        # 新しいコンテキストを作成（ビューポートサイズとフォントの設定）
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            locale='ja-JP',  # 日本語ロケールを設定
        )

        # 新しいページを開く
        page = context.new_page()

        try:
            # ページへ移動（タイムアウトを30秒に設定）
            page.goto(url, timeout=30000)

            # フォントとスタイルの設定を注入
            page.add_style_tag(content="""
                * {
                    font-family: 'Noto Sans CJK JP', 'Hiragino Kaku Gothic ProN', 'メイリオ', sans-serif !important;
                }
            """)

            # ページの読み込みが完了するまで待機
            page.wait_for_load_state('networkidle')

            # PDFとして保存
            page.pdf(
                path=output_path,
                format="A4",
                margin={
                    "top": "1cm",
                    "right": "1cm",
                    "bottom": "1cm",
                    "left": "1cm"
                },
                print_background=True
            )

            print(f"PDFを保存しました: {output_path}")

        except Exception as e:
            print(f"エラーが発生しました: {str(e)}")

        finally:
            browser.close()


if __name__ == "__main__":
    # 使用例
    url = "https://zenn.dev"  # PDFとして保存したいURLを指定
    save_webpage_as_pdf(url, "python_org.pdf")  # 出力PDFのパスを指定
