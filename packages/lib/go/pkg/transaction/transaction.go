// 雑に実装したけどリファクタリング必要だと思う
package transaction

import (
	"context"
	"errors"
)

type Transaction interface {
	context.Context
	Commit() error
	Rollback() error
	setCommit(func())
	setRollback(func())
	appendChild(Transaction)
}

type transaction struct {
	context.Context
	committed bool
	children  []Transaction
	commit    func()
	rollback  func()
}

func Begin(ctx context.Context) Transaction {
	return &transaction{
		Context:  ctx,
		commit:   func() {},
		rollback: func() {},
	}
}

func (t *transaction) appendChild(child Transaction) {
	t.children = append(t.children, child)
}

func (t *transaction) setCommit(f func()) {
	t.commit = f
}

func (t *transaction) setRollback(f func()) {
	t.rollback = f
}

func (t *transaction) Rollback() error {
	// 追加した順番と逆にrollbackする
	for i := len(t.children) - 1; i >= 0; i-- {
		t.children[i].Rollback()
	}
	t.rollback()
	return nil
}

func (t *transaction) Commit() error {
	if t.committed {
		return ErrCommitted
	}
	for _, child := range t.children {
		child.Commit()
	}
	t.commit()
	// rollbackが呼ばれてもなにもしない
	t.rollback = func() {}
	t.committed = true
	return nil
}

var (
	ErrCommitted = errors.New("transaction has already been committed")
)

func WithCommit(parent Transaction, commit func()) Transaction {
	// Contextの値を引き継ぎこれであってるかわからん
	tx := &transaction{Context: parent, commit: commit, rollback: func() {}}
	parent.appendChild(tx)
	return tx
}

func WithRollback(parent Transaction, rollback func()) Transaction {
	tx := &transaction{Context: parent, commit: func() {}, rollback: rollback}
	parent.appendChild(tx)
	return tx
}
