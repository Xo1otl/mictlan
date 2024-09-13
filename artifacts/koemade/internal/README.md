# Rules

- Graphqlリスペクトで、入力を表すEntityにはInput suffixをつける (Graphqlはクソ)
- 保存するやつと入力が全く同じに見える場合でも、idをデータベースが自動生成する場合が多いため、InputとObjectに分ける必要があることに注意する
