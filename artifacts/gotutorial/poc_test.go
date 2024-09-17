package gotutorial

import (
	"context"
	"fmt"
	"testing"
	"time"
)

func TestContextPoc(t *testing.T) {
	// 親コンテキストを作成
	parentCtx, parentCancel := context.WithCancel(context.Background())
	defer parentCancel() // この例では親コンテキストを最後にキャンセル

	// 子コンテキストを作成（親コンテキストから）
	childCtx, childCancel := context.WithCancel(parentCtx)

	// 子の子コンテキストを作成（子コンテキストから）
	grandchildCtx, _ := context.WithCancel(childCtx)

	// 各コンテキストのキャンセルを監視
	go watchContext("parentCtx", parentCtx)
	go watchContext("childCtx", childCtx)
	go watchContext("grandchildCtx", grandchildCtx)

	// 子コンテキストをキャンセル
	fmt.Println("Cancelling childCtx")
	childCancel()

	// 少し待ってから親コンテキストをキャンセル
	time.Sleep(1 * time.Second)
	fmt.Println("Cancelling parentCtx")
	parentCancel()

	// プログラムが終了しないように待機
	time.Sleep(2 * time.Second)
}

func watchContext(name string, ctx context.Context) {
	<-ctx.Done()
	fmt.Printf("%s canceled: %v\n", name, ctx.Err())
}
