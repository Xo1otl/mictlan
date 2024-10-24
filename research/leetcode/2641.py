# Definition for a binary tree node.
from typing import List, Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def replaceValueInTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root is None:
            return None
        root.val = 0

        def sums(nodes: List):
            next_nodes = []
            sum = 0

            if len(nodes) == 0:
                return
            for node in nodes:
                if node is None:
                    continue
                next_nodes.append(node.left)
                next_nodes.append(node.right)
                # その行の合計を計算
                sum += node.left.val if node.left else 0
                sum += node.right.val if node.right else 0

            for node in nodes:
                if node is None:
                    continue
                # 行の合計から自身の値を引いたものが兄弟の合計
                cousins_sum = sum
                cousins_sum -= node.left.val if node.left else 0
                cousins_sum -= node.right.val if node.right else 0
                if node.left:
                    node.left.val = cousins_sum
                if node.right:
                    node.right.val = cousins_sum

            sums(next_nodes)

        sums([root])
        return root


test_vals = [5, 4, 9, 1, 10, None, 7]


def buildTree(vals: List[int | None]) -> Optional[TreeNode]:
    val = vals.pop(0)
    root = TreeNode(val) if val is not None else None

    def build(nodes):
        next_nodes = []

        for node in nodes:
            if len(vals) == 0:
                return
            left = vals.pop(0)
            if len(vals) == 0:
                return
            right = vals.pop(0)

            if node is None:
                continue

            node.left = TreeNode(left) if left is not None else None
            node.right = TreeNode(right) if right is not None else None

            next_nodes.append(node.left)
            next_nodes.append(node.right)

        build(next_nodes)

    build([root])

    return root


tree = buildTree(test_vals)
new_tree = Solution().replaceValueInTree(tree)
print(new_tree)
