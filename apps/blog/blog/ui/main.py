import streamlit as st
import os
from streamlit.navigation.page import StreamlitPage
from util import localfile

st.set_page_config(layout="wide")


pages = {}


def add_page(root, _, files):
    section = os.path.basename(root)
    for file in files:
        if not file.endswith(".py"):
            return
        page_path = os.path.join(root, file)
        page_title = os.path.splitext(file)[0]
        relative_path = os.path.relpath(
            page_path, os.path.dirname(__file__))
        page = st.Page(relative_path, title=page_title)
        if section == "pages":
            section = ""
        if section not in pages:
            pages[section] = []
        pages[section].append(page)


localfile.walkdir("pages", add_page)


pg = st.navigation(pages)
pg.run()
