#!/bin/bash
cd /workspaces/mictlan/apps/nback
nohup bun run preview &> /workspaces/mictlan/apps/nback/out/server.log &
