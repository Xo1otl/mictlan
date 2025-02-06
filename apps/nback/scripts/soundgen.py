from gtts import gTTS

# 生成したいアルファベット（今回は A～J の10文字）
letters = ["K", "L", "M", "O"]

for letter in letters:
    # gTTS で文字を音声に変換
    tts = gTTS(text=letter, lang="en", slow=False)
    # 例として "A.mp3", "B.mp3", ... として保存
    filename = f"{letter}.mp3"
    tts.save(filename)
    print(f"{filename} saved.")
