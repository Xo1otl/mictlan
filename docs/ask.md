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
1. Read @docs/architecture.md
2. Read @apps/llmsr-worker/src/llmsr_worker/propose/_handle.py
3. Read @api/llmsr_worker/pb/llmsr.proto
3. Process @docs/task.md
   - If the "Your Task" section is present, execute its instructions.
   - If the "Question" section is present, answer its questions.
It is important that you read files in the exact order. Let's begin.
