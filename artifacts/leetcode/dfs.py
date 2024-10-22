class Solution:
    def maxUniqueSplit(self, s: str) -> int:
        res = 0  # 最大のユニークな分割数を保持する変数
        used = set()  # 使用済みの部分文字列を管理するセット

        # 現在の木の構造を保持するためのリスト
        current_tree = []

        def dfs(start, depth):
            nonlocal res  # 外側の変数 res を使用するために宣言
            indent = "  " * depth  # 再帰の深さに応じてインデントを増やす

            # 現在の元の文字列と木構造を表示
            tree_str = " -> ".join(current_tree) if current_tree else "根"
            print(f"元の文字列: '{s}' | 現在の木: {tree_str}")
            print(f"{indent}dfs(start={start}, depth={depth}) 開始")
            print(f"{indent}現在の使用済み文字列セット: {used}")

            # ベースケース: 文字列の終端に到達した場合
            if start == len(s):  # ネスト1開始
                if len(used) > res:
                    print(f"{indent}新しい最大分割数 {len(used)} を発見しました！")
                res = max(res, len(used))
                print(f"{indent}文字列の終わりに到達しました。現在の最大ユニーク分割数: {res}")
                print(f"{indent}dfs(start={start}, depth={depth}) 終了\n")
                return  # ネスト1終了

            # start から末尾まで部分文字列を生成
            for end in range(start + 1, len(s) + 1):  # ネスト2開始
                # 部分文字列を取得
                substring = s[start:end]
                print(f"{indent}部分文字列 s[{start}:{end}] = '{substring}' を試しています")

                # 部分文字列が未使用の場合
                if substring not in used:  # ネスト3開始
                    print(f"{indent}'{substring}' は未使用なので、セットに追加します")
                    used.add(substring)  # 使用済みに追加
                    current_tree.append(substring)  # 木構造に部分文字列を追加

                    # 再帰的に次の位置から探索
                    dfs(end, depth + 1)

                    # 戻ってきたら部分文字列をセットから削除（バックトラッキング）
                    used.remove(substring)
                    current_tree.pop()  # 木構造から部分文字列を削除
                    print(f"{indent}バックトラッキング: '{substring}' をセットから削除しました")
                else:
                    print(f"{indent}'{substring}' は既に使用済みのためスキップします")
                # ネスト3終了
            # ネスト2終了
            print(f"{indent}dfs(start={start}, depth={depth}) 終了\n")

        # 最初の位置から探索開始
        dfs(0, 0)

        # 最終結果を出力
        print(f"最終的な最大ユニーク分割数: {res}")
        return res

# テストケースの実行
testcase = "wwwzfvedwfvhsww"
Solution().maxUniqueSplit(testcase)
