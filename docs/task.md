# Objective
Instead of maximizing power at a single point, I want to achieve a broad, flat-top THW conversion spectrum over a defined range of wavelengths, while maintaining high efficiency.
# My Concern
構造が左右対称ならスペクトル分布も左右対称かと思ったがそうではなかった。
目標のフラットトップが左右対称のため、左右対称の構造だけ探索するようにしたいたが、きっぱり辞めたい。
普通にすべてのドメインをparamsとするように修正したい。
併せて、cleanでmodularとなるよう、code structureもドメイン全体最適化問題として簡潔にまとめたい。
# Task
refactor broaden_thg.py to adopt my concern
