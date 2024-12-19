package main

import (
	"flag"
	"zaiko/internal/apiserver"
)

func main() {
	addr := flag.String("host", "0.0.0.0:80", "bind address:port")
	flag.Parse()
	apiserver.LaunchEcho(*addr)
}
