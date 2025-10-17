## Memo
* orchestratorの実装側での型引数名の例
    ```go
    // B(asis): Proposeの入力
    // C(andidates): Proposeの主な出力
    // D(ata): Proposeの出力のうち、Observeで使わないもの
    // Q(uery): Observeの入力
    // E(vidence): Observeの出力
    ```

# Prompt
1. Read @research/qpm/src/qpm/cwes/_solver.py
2. Read @research/qpm/src/qpm/cwes/_visualize.py
3. Process @docs/task.md
   - If the "Task" section is present, execute its instructions.
   - If the "Question" section is present, answer its questions.
Let's begin.
