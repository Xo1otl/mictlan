testcase = "wwwzfvedwfvhsww"


def check_unique(s: str, binary: int):
    """
    s: 文字列
    binary: 分ける位置を表現する数（二進数の1が立っている箇所で分割）
    return: (True, 分けた数) or (False, 0)、Trueなら重複なし
    """
    # 1の位置を利用して文字列を分割
    substrs = []
    prev = 0
    for i in range(len(s) - 1):
        if binary & (1 << i):
            substrs.append(s[prev:i + 1])
            prev = i + 1
    substrs.append(s[prev:])  # 最後の部分文字列を追加

    # 重複がなければTrue、分割数を返す
    if len(set(substrs)) == len(substrs):
        return True, len(substrs)
    return False, 0


class Solution:
    def maxUniqueSplit(self, s: str) -> int:
        max_num = 0
        # すべての可能な分割をbit操作で試す
        for i in range(2 ** (len(s) - 1)):
            ok, num = check_unique(s, i)
            if ok:
                max_num = max(max_num, num)
        return max_num


# 結果の出力
print(Solution().maxUniqueSplit(testcase))
