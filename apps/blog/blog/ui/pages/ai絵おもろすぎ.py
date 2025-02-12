import streamlit as st
from workspace import path

"""
### AI絵おもろすぎ

[選択肢もデータも質問もカスタマイズできてリアルタイムで進化するアキネーター](https://akinator.mictlan.site/)を最近作りました！

文字だけで寂しかったんでイメージキャラをComfyUIとfluxで作ってみました

"""

st.image(path.Path("apps/akinator/akinator/ui/assets/akinator.png")
         .rel2(path.Path("apps/blog/blog/ui/main.py").dir()), width=400)

"""
👇promptはこれ
```
Very cute girl, mind-reading genie, open book, insightful gaze, knowing smile, ethereal glow, otherworldly aura, delicate features, flowing hair, soft light, gentle expression, mystical atmosphere, swirling energy, thought bubbles, ancient wisdom, cosmic connection, pastel colors, dreamy background, cinematic lighting, shallow depth of field, hyperrealistic
```
[workflowはこちら](https://paaster.io/d_DRKfYitbMd6-UbwU98Y#uUvDBgnUzG_nI-QxaHYXpXp6zCHMy-VRVHrRSq5cVIw)

めっちゃ可愛くていい感じですね！

Controlネットとかで姿勢制御したり、文字書いたり、手を修正したりいろいろできるみたいなんで試してみたい...🤔
"""
