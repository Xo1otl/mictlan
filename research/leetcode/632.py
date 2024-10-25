from typing import List


class Solution:
    def smallestRange(self, nums: List[List[int]]) -> List[int]:
        # すべての要素を一つのリストにまとめる
        # その際に、どのリストの要素かを記録しておく
        # すべてのリストが含まれる範囲をすべて検出する
        # 最も範囲が狭いものを返す

        merged = []
        for i, num_list in enumerate(nums):
            for n in num_list:
                merged.append((n, i))
        merged.sort(key=lambda x: x[0])

        smallest_range = 10**6
        smallest_pair = []
        for i, _ in enumerate(merged):
            # 与えられたすべてのlistsからの数を含むbatchになるまでmergedを読んでいく
            # すべての要素を含んだ場合、最初の数と最後の数の差を計算する
            # その時、smallest_rangeよりも狭い範囲が見つかった場合、それを記録する
            list_idxs = [i for i in range(len(nums))]
            first_val = merged[i][0]

            while True:
                if i >= len(merged):
                    break
                # その要素が含まれるリストのインデックスを削除、すでに削除済みなら考えなくてよい、要素がなくなったら差分を計算する
                item_idx = merged[i][1]
                if item_idx in list_idxs:
                    list_idxs.remove(item_idx)
                    if not list_idxs:
                        break
                i += 1

            if not list_idxs:
                last_val = merged[i][0]
            else:
                last_val = 10**6

            if last_val - first_val < smallest_range:
                smallest_range = last_val - first_val
                smallest_pair = [first_val, last_val]

        return smallest_pair


pair = Solution().smallestRange(
    [[10], [11]])
print(pair)
