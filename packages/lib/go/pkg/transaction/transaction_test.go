package transaction_test

import (
	"context"
	"lib/pkg/transaction"
	"strings"
	"testing"
)

func TestComplexTransaction(t *testing.T) {
	var logs []string
	logFunc := func(s string) {
		logs = append(logs, s)
	}

	testCases := []struct {
		name     string
		action   func(tx transaction.Transaction)
		expected []string
	}{
		{
			name: "Nested Commit",
			action: func(tx transaction.Transaction) {
				outerTx := transaction.WithCommit(tx, func() { logFunc("outer commit") })
				transaction.WithCommit(outerTx, func() { logFunc("inner commit 1") })
				innerTx := transaction.WithCommit(outerTx, func() { logFunc("inner commit 2") })
				transaction.WithCommit(innerTx, func() { logFunc("innermost commit") })
				tx.Commit()
			},
			expected: []string{"inner commit 1", "innermost commit", "inner commit 2", "outer commit"},
		},
		{
			name: "Mixed Commit and Rollback",
			action: func(tx transaction.Transaction) {
				outerTx := transaction.WithCommit(tx, func() { logFunc("outer commit") })
				transaction.WithRollback(outerTx, func() { logFunc("inner rollback 1") })
				innerTx := transaction.WithCommit(outerTx, func() { logFunc("inner commit") })
				transaction.WithRollback(innerTx, func() { logFunc("innermost rollback") })
				tx.Rollback()
			},
			expected: []string{"innermost rollback", "inner rollback 1"},
		},
		{
			name: "Double Commit",
			action: func(tx transaction.Transaction) {
				transaction.WithCommit(tx, func() { logFunc("commit") })
				tx.Commit()
				if err := tx.Commit(); err != transaction.ErrCommitted {
					t.Errorf("Expected ErrCommitted, got %v", err)
				}
			},
			expected: []string{"commit"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			logs = []string{}
			tx := transaction.Begin(context.Background())
			tc.action(tx)

			if !compareStringSlices(logs, tc.expected) {
				t.Errorf("Expected logs %v, got %v", tc.expected, logs)
			}
		})
	}
}

func compareStringSlices(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestTransactionRollbackOrder(t *testing.T) {
	var logs strings.Builder
	tx := transaction.Begin(context.Background())

	transaction.WithRollback(tx, func() { logs.WriteString("1") })
	innerTx := transaction.WithRollback(tx, func() { logs.WriteString("2") })
	transaction.WithRollback(innerTx, func() { logs.WriteString("3") })

	tx.Rollback()

	expected := "321"
	if logs.String() != expected {
		t.Errorf("Expected rollback order %s, got %s", expected, logs.String())
	}
}
