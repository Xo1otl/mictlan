#!/usr/bin/env node

const { join } = require("path");
const { spawnSync } = require("child_process");

const result = spawnSync(
  "python",
  [join(__dirname, "nbconvert.pex"), ...process.argv.slice(2)],
  {
    shell: true,
    stdio: "inherit",
  }
);

process.exitCode = result.status;
