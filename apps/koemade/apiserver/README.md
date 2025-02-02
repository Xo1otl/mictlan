# 声優取得API
curl "http://localhost:8001/actor/8"
```json
{
  "actor": {
    "id": "8",
    "name": "satokyosuke",
    "status": "受付中",
    "rank": "Bronze",
    "description": "隠すキャビンクルー催眠術分割サワー。知覚じぶんの緩む君は普通の。\n奨励します彼証言する人形残る。主婦コミュニティ舗装彼タワー装置ソース。\n主婦クロス暖かい教会錯覚符号障害。編組風景電池舗装欠乏再現する電池。状況錯覚合計感謝する楽しんで移動トースト。\n感謝する知覚デッド細かいバーゲンスキーム。\n販売バーゲンキャビネットクロス索引。呼ぶジャム創傷意図月オークション怒り。ノートダイヤモンド私戦略的。",
    "avator_url": "8_ジャム_8566.gif",
    "price": {
      "default": 8143,
      "nsfw": 2612,
      "nsfw_extreme": 6438
    }
  },
  "sample_voices": {
    "total_matches": 20,
    "items": [
      {
        "id": "101",
        "name": "動物それ持つ血まみれの。",
        "source_url": "8_シュガー_3376.wav",
        "tags": []
      },
// ...以下略
    ]
  }
}
```

# 音声取得API
curl "http://localhost:8001/voice/8"
```json
{
  "id": "8",
  "title": "不自然なシュガートースト電池。",
  "account": {
    "id": 3,
    "username": "kobayashimai@example.com",
    "avator_url": "3_文言_3809.gif"
  },
  "filename": "3_合計_4588.mpeg",
  "created_at": "2025-01-20 09:13:28",
  "tags": []
}
```

# 声優検索API
curl "http://localhost:8001/search-actors?nsfw_allowed=0"

```json
{
  "total_matches": 19,
  "items": [
    {
      "id": 3,
      "name": "mai99",
      "status": "受付中",
      "rank": "Diamond",
      "avatar_url": "3_文言_3809.gif"
    },
    {
      "id": 5,
      "name": "kyosukewatanabe",
      "status": "受付中",
      "rank": "Gold",
      "avatar_url": "5_持っていました_3383.png"
    },
// ...以下略
  ]
}
```

# 音声検索API
curl "http://localhost:8001/search-voices?tags[genre]=vote&tags[theme]=staff"

```json
{
  "total_matches": 7,
  "items": [
    {
      "id": "460",
      "name": "Probably indeed institution ask here.",
      "actor": {
        "id": "25",
        "name": "spencerbrandt@example.com",
        "status": "受付中",
        "rank": "Gold",
        "total_voices": 20
      },
      "tags": [
        {
          "id": 25,
          "name": "vote",
          "category": "genre"
        },
        {
          "id": 26,
          "name": "safe",
          "category": "genre"
        },
        {
          "id": 33,
          "name": "staff",
          "category": "theme"
        }
      ],
      "source_url": "25_ask_3565.ogg"
    },
    {
      "id": "491",
      "name": "Could bank practice dinner.",
      "actor": {
        "id": "27",
        "name": "mikaylabright@example.com",
        "status": "受付中",
        "rank": "Gold",
        "total_voices": 20
      },
      "tags": [
        {
          "id": 25,
          "name": "vote",
          "category": "genre"
        },
        {
          "id": 29,
          "name": "and",
          "category": "mood"
        },
        {
          "id": 33,
          "name": "staff",
          "category": "theme"
        }
      ],
      "source_url": "27_there_7019.wav"
    },
// ...以下略
  ]
}
```

検索クエリ例(パラメータ全部指定版)
# 音声検索API
curl "http://localhost:8001/search-voices?title=bank&tags[genre]=vote&tags[theme]=staff&page=1"

```json
{
  "total_matches": 1,
  "items": [
    {
      "id": "491",
      "name": "Could bank practice dinner.",
      "actor": {
        "id": "27",
        "name": "mikaylabright@example.com",
        "status": "受付中",
        "rank": "Gold",
        "total_voices": 20
      },
      "tags": [
        {
          "id": 25,
          "name": "vote",
          "category": "genre"
        },
        {
          "id": 29,
          "name": "and",
          "category": "mood"
        },
        {
          "id": 33,
          "name": "staff",
          "category": "theme"
        }
      ],
      "source_url": "27_there_7019.wav"
    }
  ]
}
```

# 声優検索API
curl "http://localhost:8001/search-actors?nsfw_allowed=0&name_like=ale&status=%E5%8F%97%E4%BB%98%E4%B8%AD&page=1&nsfw_extreme_allowed=0"

```json
{
  "total_matches": 1,
  "items": [
    {
      "id": 42,
      "name": "alevy",
      "status": "受付中",
      "rank": "Silver",
      "avatar_url": "42_call_8215.png"
    }
  ]
}
```
