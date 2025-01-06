### 1. サンプルボイス検索用API

`curl "http://localhost:8001/search-voices?title=%E3%83%95%E3%82%A7%E3%83%9F%E3%83%8B%E3%82%B9%E3%83%88&tags%5B%5D=mood:%E7%8A%AF%E7%BD%AA%E8%80%85&page=1"`

**リクエスト:**
- `title`: タイトル
- `tags`: tagの配列 (カテゴリと名前、性別、年齢、deliveryなど)
- `page`: Page number for pagination (e.g., 1)

**レスポンス:**
```json
{
    "total": 120,
    "page": 1,
    "per_page": 20,
    "voices": [
        {
            "id": 101,
            "name": "Happy Greeting",
            "source_url": "https://koemade.net/samples/happy_greeting.mp3",
            "tags": ["greeting", "happy", "female"],
            "actor": {
                "id": 10,
                "name": "Jane Doe",
                "status": "enabled",
                "rank": "senior",
                "total_voices": 50
            }
        },
        {
            "id": 102,
            "name": "Excited Announcement",
            "source_url": "https://koemade.net/samples/excited_announcement.mp3",
            "tags": ["announcement", "excited", "female"],
            "actor": {
                "id": 10,
                "name": "Jane Doe",
                "status": "enabled",
                "rank": "senior",
                "total_voices": 50
            }
        },
        // More voice entries...
    ]
}
```

### 2. 声優検索用API

`curl "http://localhost:8001/search-actors?name_like=&status=%E5%8F%97%E4%BB%98%E4%B8%AD&page=1"`

条件を満たすサンプルボイスを投稿したユーザーを検索できるのが理想

**リクエスト:**
- `name_like`: 名前
- `status`: 受付中と受付停止中
- `nsfw_options`: nsfwの設定でokかextreme.okかどうか
- `page`: ページ番号

**レスポンス:**
```json
{
    "total": 50,
    "page": 1,
    "per_page": 20,
    "actors": [
        {
            "id": 20,
            "name": "John Smith",
            "status": "enabled",
            "rank": "expert",
            "avatar_url": "https://koemade.net/avatars/john_smith.jpg"
        },
        {
            "id": 21,
            "name": "John Doe",
            "status": "enabled",
            "rank": "intermediate",
            "avatar_url": "https://koemade.net/avatars/john_doe.jpg"
        },
        // More actor entries...
    ]
}
```

### 3. 声優プロフィールAPI

`curl "http://localhost:8001/actor/10"`

nsfwが許可されていない場合は適当にfalse入れて返す

**リクエスト:**
- `id`: 声優のID
- `page`: ページ番号

**レスポンス:**
```json
{
    "actor": {
        "id": 20,
        "name": "John Smith",
        "status": "enabled",
        "rank": "expert",
        "description": "Experienced voice actor with a deep and soothing voice.",
        "avatar_url": "https://koemade.net/avatars/john_smith.jpg",
        "price": {
            "default": 100,
            "nsfw": 150,
            "nsfw_extreme": 250
        }
    },
    "sample_voices": [
        {
            "id": 110,
            "name": "Narration Sample",
            "source_url": "https://koemade.net/samples/narration_sample.mp3",
            "tags": ["narration", "deep", "male"]
        },
        {
            "id": 111,
            "name": "Character Voice",
            "source_url": "https://koemade.net/samples/character_voice.mp3",
            "tags": ["character", "mysterious", "male"]
        },
        // More sample voice entries...
    ]
}
```

### 4. ボイス詳細API

**リクエスト:**
- `id`: ボイスID (例: 101)

**レスポンス:**
```json
{
    "voice": {
        "id": 1,
        "title": "サンプルボイス",
        "account": {
            "id": 2,
            "username": "qlovolp.ttt@gmail.com",
            "avator_url": "abcd@example.com"
        },
        "mime_type": "audio/mpeg",
        "filename": "sample.mp3",
        "created_at": "2023-10-27 15:30:00",
        "tags": [
            {
                "id": 1,
                "tag_name": "10代",
                "tag_category": "年代別タグ"
            },
            {
                "id": 5,
                "tag_name": "快活",
                "tag_category": "キャラ別タグ"
            }
        ]
    }
}
```
