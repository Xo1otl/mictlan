import streamlit as st

st.set_page_config(layout="wide")

"""
# Monorepoについて

無限の拡張性を考える時、再帰的に考えるのが一番いいと思っています

私が作るリポジトリの構成は基本以下のような感じです

```
package_name
├── src/ (internalやpackage_nameの場合もある)
├── cmd/
├── docs/
├── scripts/
├── projectfile
```

srcの中にドメインとアダプタを書いています
"""
