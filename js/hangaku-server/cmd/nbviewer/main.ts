import * as elysia from "../../internal/elysia";

// コマンドライン引数からポート番号を取得する関数
function getPortFromArgs(): number {
  const args = process.argv.slice(2);
  let port = 4002; // デフォルト値

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--port" && i + 1 < args.length) {
      const portValue = Number.parseInt(args[i + 1], 10);
      if (!Number.isNaN(portValue)) {
        port = portValue;
        break;
      }
    }
  }

  return port;
}

// ポート番号を取得
const port = getPortFromArgs();

// Elysiaを起動
elysia.launchNbviewer(port);

console.log(`Listening on http://localhost:${port} ...`);
