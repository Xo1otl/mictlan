class Tensor:
    def __init__(self, dim):
        self.dim = dim
        self.fixed_dims = [None] * dim
        self.data = {}

    def __setitem__(self, index, value):
        # TODO: dimの範囲内かどうかをチェック
        self.data[index] = value

    def __getitem__(self, index):
        if isinstance(index, int):
            index = (index,)

        index_iter = iter(index)

        # fixed_dimsと結合して取得すべきindexを作成
        index = [
            next(index_iter) if fixed is None else fixed
            for fixed in self.fixed_dims
        ]

        # 完全なindexを持っていない場合viewを返す
        if any(i is None for i in index):
            view = Tensor(self.dim)
            view.data = self.data
            view.fixed_dims = index
            return view

        # 完全なindexを持っている場合値を返す
        return self.data[tuple(index)]

    def __repr__(self):
        return f"Tensor(dim={self.dim}, fixed_dims={self.fixed_dims}, data={self.data})"


# テスト
tensor = Tensor(4)
tensor[0, 0, 0, 0] = 1
tensor[1, 3, 1, 0] = 2
tensor[1, 3, 1, 1] = 3
view = tensor[1, None, None, None]  # viewは二次元のテンソルに見える、[3,1]に2が入っている
for i in view[3, 1, None]:
    print(i)
