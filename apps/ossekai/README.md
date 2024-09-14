# ossekaiserver

gin で作る

# Coding Style

機能毎に module 作った後、interface adapter の実装はその module の中に作る

# Note

- sub (Subject): ユーザーの一意識別子

  - JWT 標準クレームの一つで、OIDC にも準拠
  - アプリケーション層全体で一貫して使用
  - データベースのプライマリーキーやユーザー参照に利用
  - 例: User.sub, Post.author_sub
  - 抽象的で汎用的な概念のため、認証システムの変更にも柔軟に対応可能

- アプリケーション層の抽象概念:
  - Principal: 認証されたエンティティ（ユーザーやシステム）を表す
  - Subject: Principal と同様、認証されたエンティティを指す。JWT では sub クレームとして使用
  - Identity: ユーザーやエンティティの識別情報を抽象化した概念
  - Claims: エンティティに関する追加情報や属性のセット

# Memo

GetUser 関数で JWT の検証やってくれるらしい

jwt だと少し高速化するらしい

# TODO

- commandserverとqueryserverを分けて二つのcmdを用意し、CQRSを行う
- commandserverは今作られているような感じ
- queryserverでは複雑な検索の対応を考える

# データの大体の形式

```json
{
	"Questions": [
		{
			"Sub": "68230afe-eec9-46a5-afe3-b38c7bd1d5e7",
			"Id": "9aaf39d7-629d-4c2f-adb1-b1b6ef21560c",
			"Title": "Behind unemployment sit anybody fascinate.",
			"CreatedAt": "2024-09-14T14:53:50.836392268Z",
			"UpdatedAt": "2024-09-14T14:53:50.836392338Z",
			"BestAnswerId": "",
			"Tags": [
				{ "Id": "a1062d23-bef2-4255-a2b1-19738a41167f", "Name": "Squeak" }
			],
			"ContentBlocks": [
				{
					"Kind": "markdown",
					"Content": "{{Chimpanzeemay}} Est eum corrupti maxime fugit consequuntur commodi fugiat repudiandae quibusdam mollitia repellat a molestias nihil labore voluptas veritatis at eius voluptatibus sit numquam quaerat nostrum esse ex eaque et qui occaecati dolores debitis perspiciatis est id et eum ut ut nostrum suscipit reiciendis voluptatem hic voluptate alias reprehenderit amet ex."
				},
				{
					"Kind": "text",
					"Content": "So otherwise such yours then does theirs any as would. Trade dynasty then here luxury huh her its massage words. Data yours themselves mushy solemnly that far anybody have other."
				},
				{
					"Kind": "text",
					"Content": "{{GreenYellowanswer}} our that recently cheese many everything anyone yearly she. Annually beneath fortnightly hourly Turkish range then staff shiny a. Of to they clean us someone you over what balloon."
				},
				{
					"Kind": "text",
					"Content": "Whoever also here whom how the the hmm life move. As it of care mustering might listen would here easy. What down strongly roll these goodness man mob brightly reel."
				},
				{
					"Kind": "latex",
					"Content": "Qui id ut molestiae beatae inventore harum facilis possimus neque recusandae debitis ab ipsam labore modi excepturi sit inventore officia fugiat maxime dolorem voluptatem esse id perspiciatis nemo ea aut velit harum debitis nihil in animi rem quo quia nostrum quis aut omnis odit magnam aut mollitia hic eum cumque."
				}
			],
			"Attachments": [
				{
					"Placeholder": "Chimpanzeemay",
					"Kind": "application/octet-stream",
					"Size": 353,
					"ObjectKey": "b4dece02-9404-4a03-ad17-a1ae7b10fe6e"
				},
				{
					"Placeholder": "GreenYellowanswer",
					"Kind": "application/octet-stream",
					"Size": 344,
					"ObjectKey": "56a90983-2b01-42ed-9d9b-55b715e0214a"
				}
			]
		},
		{
			"Sub": "99e37f0a-7e4c-48bb-86d9-2c74598cd266",
			"Id": "5a767fc6-3859-46c9-b8dd-5b5967392602",
			"Title": "Meanwhile that tent for British.",
			"CreatedAt": "2024-09-14T15:07:38.008655048Z",
			"UpdatedAt": "2024-09-14T15:07:38.008655108Z",
			"BestAnswerId": "",
			"Tags": [
				{ "Id": "0ba1089f-a1a2-40af-ab87-cfc943f3dce3", "Name": "MSL" },
				{ "Id": "c5b4f874-d90c-4903-a10c-866d33e4b059", "Name": "Vala" }
			],
			"ContentBlocks": [
				{
					"Kind": "markdown",
					"Content": "Assumenda minus dolor {{LightSalmonmoonlight}} explicabo alias voluptas et laborum maiores non molestiae est quod {{Olivehail}} assumenda est dolore quisquam quibusdam esse consequatur qui suscipit voluptate eum corporis commodi eaque quia quam et ducimus sunt vero cum pariatur explicabo possimus vel sit voluptate quibusdam expedita porro natus aut consectetur rem aut."
				}
			],
			"Attachments": [
				{
					"Placeholder": "LightSalmonmoonlight",
					"Kind": "application/octet-stream",
					"Size": 342,
					"ObjectKey": "4184cddc-2c99-4969-b443-2306d159de47"
				},
				{
					"Placeholder": "Olivehail",
					"Kind": "application/octet-stream",
					"Size": 352,
					"ObjectKey": "a002fcf0-e69e-4d7d-9b6e-efee2abec583"
				}
			]
		}
	]
}
```
