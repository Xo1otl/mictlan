# mail

gmail使ってんのダサいから自分で建てる

## refs
- Aレコードを設定してMXレコードを設定してdkimを設定してdmarcを設定
```
xolotl@DESKTOP-KUKIJ4R:/mnt/c/Users/Cairo$ dig _dmarc.mictlan.site TXT

; <<>> DiG 9.18.28-0ubuntu0.22.04.1-Ubuntu <<>> _dmarc.mictlan.site TXT
;; global options: +cmd
;; Got answer:
;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: 18319
;; flags: qr rd ra ad; QUERY: 1, ANSWER: 1, AUTHORITY: 0, ADDITIONAL: 1

;; OPT PSEUDOSECTION:
; EDNS: version: 0, flags:; udp: 1410
;; QUESTION SECTION:
;_dmarc.mictlan.site.           IN      TXT

;; ANSWER SECTION:
_dmarc.mictlan.site.    1799    IN      TXT     "v=DMARC1; p=none"

;; Query time: 50 msec
;; SERVER: 10.255.255.254#53(10.255.255.254) (UDP)
;; WHEN: Fri Oct 11 22:04:06 JST 2024
;; MSG SIZE  rcvd: 77

xolotl@DESKTOP-KUKIJ4R:/mnt/c/Users/Cairo$ dig mail._domainkey.mictlan.site TXT

; <<>> DiG 9.18.28-0ubuntu0.22.04.1-Ubuntu <<>> mail._domainkey.mictlan.site TXT
;; global options: +cmd
;; Got answer:
;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: 42564
;; flags: qr rd ra ad; QUERY: 1, ANSWER: 1, AUTHORITY: 0, ADDITIONAL: 1

;; OPT PSEUDOSECTION:
; EDNS: version: 0, flags:; udp: 1410
;; QUESTION SECTION:
;mail._domainkey.mictlan.site.  IN      TXT

;; ANSWER SECTION:
mail._domainkey.mictlan.site. 1632 IN   TXT     "v=DKIM1; k=rsa; p=MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAtXSJ3Ab4wN/zgEAVjPnCxGAmAC+tcRz4pUQ0qaPwo5gcdb2202AA66LKiysiSThzUdtb9aKSK+VPFQShS7MO6TlSL09RbzjVxYFpAapEF2isEa3eHajX+h8GEuqLTRLe5Xal0QGTAXkeCiIlMf0lgtxCOZQmUfwKChYidvNyWblCgKr3WTFcvBDeqpRslBjfK" "CtrEXFIt98kkMgkvN74oxFnja1cTqvRSPvsZ0rvmM5lNvMwzxyHcgtYPokEEanpy4IsuWYn7ZTvdsAdMHpAuB4iYIyx+EGP2KCmP8L/ax9khP/k7IVc3dVOuN/SuO9ZhfXlu7bk1cyjgY+m9PaWrwIDAQAB"

;; Query time: 10 msec
;; SERVER: 10.255.255.254#53(10.255.255.254) (UDP)
;; WHEN: Fri Oct 11 22:04:15 JST 2024
;; MSG SIZE  rcvd: 481

xolotl@DESKTOP-KUKIJ4R:/mnt/c/Users/Cairo$ dig mail.mictlan.site MX

; <<>> DiG 9.18.28-0ubuntu0.22.04.1-Ubuntu <<>> mail.mictlan.site MX
;; global options: +cmd
;; Got answer:
;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: 14018
;; flags: qr rd ra ad; QUERY: 1, ANSWER: 0, AUTHORITY: 1, ADDITIONAL: 1

;; OPT PSEUDOSECTION:
; EDNS: version: 0, flags:; udp: 1410
;; QUESTION SECTION:
;mail.mictlan.site.             IN      MX

;; AUTHORITY SECTION:
mictlan.site.           2533    IN      SOA     dns1.registrar-servers.com. hostmaster.registrar-servers.com. 1728631483 43200 3600 604800 3601

;; Query time: 10 msec
;; SERVER: 10.255.255.254#53(10.255.255.254) (UDP)
;; WHEN: Fri Oct 11 22:04:19 JST 2024
;; MSG SIZE  rcvd: 119

xolotl@DESKTOP-KUKIJ4R:/mnt/c/Users/Cairo$ dig mail.mictlan.site

; <<>> DiG 9.18.28-0ubuntu0.22.04.1-Ubuntu <<>> mail.mictlan.site
;; global options: +cmd
;; Got answer:
;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: 23034
;; flags: qr rd ra ad; QUERY: 1, ANSWER: 1, AUTHORITY: 0, ADDITIONAL: 1

;; OPT PSEUDOSECTION:
; EDNS: version: 0, flags:; udp: 1410
;; QUESTION SECTION:
;mail.mictlan.site.             IN      A

;; ANSWER SECTION:
mail.mictlan.site.      623     IN      A       150.230.62.123

;; Query time: 0 msec
;; SERVER: 10.255.255.254#53(10.255.255.254) (UDP)
;; WHEN: Fri Oct 11 22:05:42 JST 2024
;; MSG SIZE  rcvd: 62

```
- [dmarcについて](https://www.cloudflare.com/learning/email-security/dmarc-dkim-spf/)
- リレーがgmailになっているので、そのままだと送信は実質gmailから行われる
- gmailで二段階認証を有効化してから[relay password(アプリパスワード)](https://support.google.com/mail/answer/185833?hl=ja)を設定
- リレーのgmailでのFromの反映方法
    - gmailのsee all settingsからAccounts and Importを選んでsend mail asを追加
    - defaultにmictlan.siteのメールアドレスを指定してtreat as an aliasを指定した状態で設定
