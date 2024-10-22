from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def kthLargestLevelSum(self, root: Optional[TreeNode], k: int) -> int:
        sums = []

        def dfs(node, level):
            if not node:
                return
            if len(sums) < level:
                sums.append(0)
            sums[level - 1] += node.val
            dfs(node.left, level + 1)
            dfs(node.right, level + 1)

        dfs(root, 1)

        sums.sort(reverse=True)
        return sums[k-1]


demo = TreeNode(val=5, left=TreeNode(val=8, left=TreeNode(val=2, left=TreeNode(val=4, left=None, right=None), right=TreeNode(val=6, left=None, right=None)), right=TreeNode(
    val=1, left=None, right=None)), right=TreeNode(val=9, left=TreeNode(val=3, left=None, right=None), right=TreeNode(val=7, left=None, right=None)))

s = Solution().kthLargestLevelSum(demo, 2)
print(s)
