from typing import List


class Solution:
    # TODO: すべてのブロックの辞書を作る
    # それぞれのブロックについて以下の情報を持つ
    # row_neibors, col_neibors
    # ブロックの数 * 4 - 隣接するブロックの数
    # 28 - 12 = 16
    # 1 + 1 + 4 + 1 + 2 + 2 + 1
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        grid_num = 0
        conflicts = 0
        for r_i, row in enumerate(grid):
            for c_i, col in enumerate(row):
                if grid[r_i][c_i] == 1:
                    grid_num += 1
                if col == 1:
                    if r_i - 1 >= 0 and grid[r_i - 1][c_i] == 1:
                        conflicts += 1
                    if r_i + 1 < len(grid) and grid[r_i + 1][c_i] == 1:
                        conflicts += 1
                    if c_i - 1 >= 0 and grid[r_i][c_i - 1] == 1:
                        conflicts += 1
                    if c_i + 1 < len(row) and grid[r_i][c_i + 1] == 1:
                        conflicts += 1
        return 4 * grid_num - conflicts
