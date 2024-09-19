package gotutorial

import "testing"

type Node interface {
	Func()
}

type node struct {
	Node
	children []Node
	f        func()
}

func New() Node {
	return &node{children: nil, f: nil}
}

// Func implements Node.
func (n *node) Func() {
	for _, c := range n.children {
		c.Func()
	}
	if n.f != nil {
		n.f()
	}
}

func propagate(parent Node, child node) {
	pn := &node{parent, nil, nil}
	pn.children = append(pn.children, &child)
}

func WithChild(parent Node) Node {
	n := &node{parent, nil, nil}
	propagate(parent, *n)
	return n
}

func Test(t *testing.T) {
	root := New()
	WithChild(root)
	WithChild(root)
	WithChild(root)
	WithChild(root)
	WithChild(root)
	WithChild(root)
	WithChild(root)
	root.Func()
}
