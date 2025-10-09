## Memo
* 推敲の時に、`Please remove any modifiers or exaggerations and revise your writing to focus on describing only the "how" and "what" necessary to reproduce the logic.`を追記するといい感じになる。GEMINI.mdでこの手順追加しようかな
* orchestratorの実装側での型引数名の例
    ```go
    // B(asis): Proposeの入力
    // C(andidates): Proposeの主な出力
    // D(ata): Proposeの出力のうち、Observeで使わないもの
    // Q(uery): Observeの入力
    // E(vidence): Observeの出力
    ```

# Prompt
1. Read @docs/README.md
2. Process @apps/orchestrator/docs/task.md:
   - If the "Your Task" section is present, execute its instructions.
   - If the "Question" section is present, answer its questions.
It is important that you read files in exact order. Let's begin.
