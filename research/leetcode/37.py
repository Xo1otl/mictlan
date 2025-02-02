import pandas as pd
import z3


test_data = [
    ["5", "3", ".", ".", "7", ".", ".", ".", "."],
    ["6", ".", ".", "1", "9", "5", ".", ".", "."],
    [".", "9", "8", ".", ".", ".", ".", "6", "."],
    ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
    ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
    ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
    [".", "6", ".", ".", ".", ".", "2", "8", "."],
    [".", ".", ".", "4", "1", "9", ".", ".", "5"],
    [".", ".", ".", ".", "8", ".", ".", "7", "9"]
]

data = []
for r in range(9):
    for c in range(9):
        g = 3 * (r // 3) + (c // 3)
        data.append({"row": r, "col": c, "grid": g,
                    "value": z3.Int(f"x_{r}_{c}")})

df = pd.DataFrame(data)

solver = z3.Solver()

for r in range(9):
    for c in range(9):
        value = test_data[r][c]
        if value != ".":
            solver.add(df.loc[(df["row"] == r) & (
                df["col"] == c), "value"].iloc[0] == int(value))

for cell in df["value"]:
    solver.add(cell >= 1, cell <= 9)

for i in range(9):
    solver.add(z3.Distinct(df.loc[df["row"] == i, "value"].tolist()))  # 行
    solver.add(z3.Distinct(df.loc[df["col"] == i, "value"].tolist()))  # 列
    solver.add(z3.Distinct(df.loc[df["grid"] == i, "value"].tolist()))  # グリッド

if solver.check() == z3.sat:
    model = solver.model()
    df["value"] = [model[cell] for cell in df["value"]]
    print("解が見つかりました:")
    print(df.pivot(index="row", columns="col", values="value"))
else:
    print("解が見つかりませんでした。")
