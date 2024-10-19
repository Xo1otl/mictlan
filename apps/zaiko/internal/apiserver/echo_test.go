package apiserver_test

import (
	"testing"
	"zaiko/internal/apiserver"
)

func TestLaunchEcho(t *testing.T) {
	apiserver.LaunchEcho("localhost:3030")
}
