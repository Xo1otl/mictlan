### 1. Sample Voice Search API

**Endpoint:**
```
GET https://api.koemade.net/voices
```

**Request Parameters:**

- `keyword`: Search term (e.g., "happy")
- `status`: Status of acceptance (any, enabled, disabled)
- `sex`: Voice actor's gender (any, male, female)
- `rating`: Content rating (any, default, r18, extreme)
- `age`: Age group of the voice (any, 0: kids, 1: teens, 2: twenties, 3: thirties, 7: forties and above)
- `delivery`: Character traits of the voice (any, 4: quiet, 5: lively, 6: sexy/mature, 8: young, 9: serious, 10: other)
- `page`: Page number for pagination (e.g., 1)

**Example Request URL:**
```
https://api.koemade.net/voices?keyword=happy&status=enabled&sex=female&rating=default&age=2&delivery=5&page=1
```

**Response Example:**

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
            "ratings": {
                "overall": 4.5,
                "clarity": 4.2,
                "naturalness": 4.8
            },
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
            "ratings": {
                "overall": 4.7,
                "clarity": 4.5,
                "naturalness": 4.9
            },
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

### 2. Actor Search API

**Endpoint:**
```
GET https://api.koemade.net/actors
```

**Request Parameters:**

- `keyword`: Search term (e.g., "John")
- `status`: Status of acceptance (any, enabled, disabled)
- `sex`: Voice actor's gender (any, male, female)
- `rating`: Content rating (any, default, r18, extreme)
- `age`: Age group of the voice (any, 0: kids, 1: teens, 2: twenties, 3: thirties, 7: forties and above)
- `delivery`: Character traits of the voice (any, 4: quiet, 5: lively, 6: sexy/mature, 8: young, 9: serious, 10: other)
- `page`: Page number for pagination (e.g., 1)

**Example Request URL:**
```
https://api.koemade.net/actors?keyword=John&status=enabled&sex=male&rating=default&age=3&delivery=9&page=1
```

**Response Example:**

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

### 3. Actor Profile API

**Endpoint:**
```
GET https://api.koemade.net/actor/
```

**Request Parameters:**

- `id`: Actor's ID (e.g., 20)
- `page`: Page number for sample voices (e.g., 1)

**Example Request URL:**
```
https://api.koemade.net/actor/?id=20&page=1
```

**Response Example:**

```json
{
    "actor": {
        "id": 20,
        "name": "John Smith",
        "status": "enabled",
        "rank": "expert",
        "description": "Experienced voice actor with a deep and soothing voice.",
        "avatar_url": "https://koemade.net/avatars/john_smith.jpg",
        "pricing": {
            "base_rate": 100,
            "additional_options": {
                "urgent_delivery": 50,
                "special_effects": 30
            }
        }
    },
    "sample_voices": [
        {
            "id": 110,
            "name": "Narration Sample",
            "source_url": "https://koemade.net/samples/narration_sample.mp3",
            "tags": ["narration", "deep", "male"],
            "ratings": {
                "overall": 4.6,
                "clarity": 4.4,
                "naturalness": 4.8
            }
        },
        {
            "id": 111,
            "name": "Character Voice",
            "source_url": "https://koemade.net/samples/character_voice.mp3",
            "tags": ["character", "mysterious", "male"],
            "ratings": {
                "overall": 4.5,
                "clarity": 4.3,
                "naturalness": 4.7
            }
        },
        // More sample voice entries...
    ]
}
```

### 4. Voice Details API

**Endpoint:**
```
GET https://api.koemade.net/voice/
```

**Request Parameters:**

- `id`: Voice ID (e.g., 101)

**Example Request URL:**
```
https://api.koemade.net/voice/?id=101
```

**Response Example:**

```json
{
    "voice": {
        "id": 101,
        "name": "Happy Greeting",
        "source_url": "https://koemade.net/samples/happy_greeting.mp3",
        "tags": ["greeting", "happy", "female"],
        "ratings": {
            "overall": 4.5,
            "clarity": 4.2,
            "naturalness": 4.8
        },
        "actor": {
            "id": 10,
            "name": "Jane Doe",
            "status": "enabled",
            "rank": "senior",
            "total_voices": 50
        },
        "transcript": "Hello everyone, have a wonderful day!",
        "duration_seconds": 5.2,
        "pricing": {
            "base_price": 75,
            "license_type": "royalty-free"
        }
    }
}
```
