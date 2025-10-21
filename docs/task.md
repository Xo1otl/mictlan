# Objective
Instead of maximizing power at a single point, I want to achieve a broad, flat-top THW conversion spectrum over a defined range of wavelengths, while maintaining high efficiency.
# My Concern
変換効率のoptimizeの時に用いた正則化項はフラットトップ広帯域のoptimizationでは役に立たないと思われる。
コードから完全に除去したい。
また、今後の拡張性のためにも、cleanでmodularなcode structureにidiomaticな方法で、リファクタリングできる部分があればリファクタリングしてほしい。
冗長性を無くし、コンパクトさ、シンプルさを大事にすること。
# Task
refactor broaden_thg.py to adopt my concern
