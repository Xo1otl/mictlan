package transaction

import (
	"context"
	"errors"
)

type Transaction interface {
	context.Context
	Commit() error
	Rollback() error
	Propagate(Transaction)
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
		Context: ctx,
	}
}

func (t *transaction) Propagate(child Transaction) {
	t.children = append(t.children, child)
}

func (t *transaction) Rollback() error {
	// 追加した順番と逆にrollbackする
	for i := len(t.children) - 1; i >= 0; i-- {
		t.children[i].Rollback()
	}
	if t.rollback != nil {
		t.rollback()
	}
	return nil
}

func (t *transaction) Commit() error {
	if t.committed {
		return ErrCommitted
	}
	for _, child := range t.children {
		child.Commit()
	}
	if t.commit != nil {
		t.commit()
	}
	t.rollback = nil
	t.committed = true
	return nil
}

var (
	ErrCommitted = errors.New("transaction has already been committed")
)

func WithCommit(parent Transaction, commit func()) Transaction {
	// Contextの値を引き継ぎこれであってるかわからん
	tx := &transaction{Context: parent, commit: commit}
	parent.Propagate(tx)
	return tx
}

func WithRollback(parent Transaction, rollback func()) Transaction {
	tx := &transaction{Context: parent, rollback: rollback}
	parent.Propagate(tx)
	return tx
}
