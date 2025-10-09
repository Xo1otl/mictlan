# mictlan

## 環境構築
1. `git clone git@github.com:Xo1otl/mictlan.git`する
2. `secrets.tar.gz`をproject rootに配置する
3. vscodeの指示に従い、open in containerする

## windowsの設定
* `New-NetFirewallRule -DisplayName "Allow Cloudflare WARP Inbound" -Direction Inbound -Action Allow -RemoteAddress "100.96.0.0/12"`
