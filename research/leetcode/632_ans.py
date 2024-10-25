from typing import List
import heapq


class Solution:
    def smallestRange(self, nums: List[List[int]]) -> List[int]:
        """
        k個のソート済みリストから、各リストから少なくとも1つの要素を含む最小の範囲を見つける

        アプローチ:
        1. 各リストの先頭の要素をヒープに入れる
        2. 現在のヒープ内の最小値と最大値で範囲を作る
        3. 最小値を出して、同じリストの次の要素を入れる
        4. これを繰り返して、最小の範囲を見つける
        """

        # デバッグ用の状態表示関数
        def print_state(heap, cur_max, small, step_name):
            print(f"\n=== {step_name} ===")
            print(f"現在のヒープ: {sorted(heap)}")  # ヒープの中身を見やすくするためソート
            print(f"現在の最大値: {cur_max}")
            print(f"現在の最小範囲: {small}")

        heap = []  # 要素は (値, リストのインデックス, リスト内での位置)
        cur_max = float('-inf')  # 現在の範囲の最大値

        # 1. 初期化: 各リストの最初の要素をヒープに追加
        # print("\n=== 初期状態 ===")
        # print(f"入力リスト: {nums}")

        for i in range(len(nums)):
            val = nums[i][0]  # i番目のリストの最初の値
            heapq.heappush(heap, (val, i, 0))  # (値, リストのインデックス, リスト内位置)
            cur_max = max(cur_max, val)  # 最大値を更新
            # print(f"リスト{i}の先頭要素{val}をヒープに追加。現在の最大値: {cur_max}")

        # 最小範囲を保持する変数を初期化
        small = [float('-inf'), float('inf')]
        # print_state(heap, cur_max, small, "初期化完了")

        step = 1
        # 2. メインループ: ヒープが空になるまで続ける
        while heap:
            # ヒープから最小値を取り出す
            cur_min, list_idx, i = heapq.heappop(heap)
            # print(f"\n=== ステップ {step} ===")
            # print(f"取り出した要素: 値={cur_min}, リスト={list_idx}, 位置={i}")

            # 現在の範囲(cur_min から cur_max)が、これまでの最小範囲より小さければ更新
            if cur_max - cur_min < small[1] - small[0]:
                small = [cur_min, cur_max]
                # print(
                #     f"新しい最小範囲を発見: [{cur_min}, {cur_max}], 幅: {cur_max - cur_min}")

            # 同じリストの次の要素があれば、ヒープに追加
            if i + 1 < len(nums[list_idx]):
                nxt = nums[list_idx][i + 1]
                heapq.heappush(heap, (nxt, list_idx, i+1))
                # 新しい要素が現在の最大値より大きければ更新
                old_max = cur_max
                cur_max = max(cur_max, nxt)
                # if cur_max != old_max:
                    # print(f"最大値を{old_max}から{cur_max}に更新")
                # print(f"リスト{list_idx}の次の要素{nxt}をヒープに追加")
            else:
                # print(f"リスト{list_idx}の要素をすべて使用済み。終了します。")
                break

            # print_state(heap, cur_max, small, f"ステップ{step}完了")
            step += 1

        # print("\n=== 最終結果 ===")
        # print(f"最小範囲: {small}")
        return small # type: ignore


# テスト用のコード
test_cases = [
    [[4, 10, 15, 24, 26], [0, 9, 12, 20], [5, 18, 22, 30]],
    [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
]

solution = Solution()
for i, test_case in enumerate(test_cases):
    print(f"\n============ テストケース {i+1} ============")
    result = solution.smallestRange(test_case)
    print(f"結果: {result}")
