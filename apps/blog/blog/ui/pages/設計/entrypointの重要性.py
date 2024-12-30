import streamlit as st

"""
# entrypointの重要性

entrypointって重要だなと思うことが多々あるのでメモ

例えばlayoutだったりmiddlewareだったりを、すべてのページの先頭に書くような方式は、責務の分離もできていない(すべてのファイルがmiddlewareの存在を知っている)

このような場合にentrypointを用意できるのは重要である

streamlitでlayout="wide"を指定する部分や、phpで書いたmiddlewareを毎回すべてのページでimportしようとするなどの失敗をした

このような場合、必ずentrypointを用意する方法を調べるべき

verticalなモジュール・entrypoint・middleware・adapterなどから構成するのがよさそう
"""
