import mysql.connector
from faker import Faker
import random
import hashlib
from infra import koemade
import bcrypt

# fake = Faker('ja_JP')
fake = Faker()

# MySQL Database Configuration
db_config = {
    "host": "mysql",
    "user": koemade.MYSQL_USER,
    "password": koemade.MYSQL_PASSWORD,
    "database": koemade.MYSQL_DB
}

DEFAULT_PASSWORD = "password123"  # Default password
HASHED_PASSWORD = bcrypt.hashpw(
    DEFAULT_PASSWORD.encode(), bcrypt.gensalt()).decode()


def insert_data(table_name, data, exclude_columns=None):
    if exclude_columns is None:
        exclude_columns = []
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        columns = [col for col in data.keys() if col not in exclude_columns]
        placeholders = ', '.join(['%s'] * len(columns))
        sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
        cursor.execute(sql, [data[col] for col in columns])
        conn.commit()
        inserted_id = cursor.lastrowid
        cursor.close()
        conn.close()
        return inserted_id
    except mysql.connector.Error as err:
        print(f"Error inserting data into {table_name}: {err}")
        return None


def generate_account():
    username = fake.unique.user_name() + "@example.com"
    return {
        "username": username,
        "password": HASHED_PASSWORD
    }


def generate_signup_request(account_id):
    return {
        "name": fake.name(),
        "furigana": fake.kana_name(),
        "address": fake.address(),
        "email": fake.email(),
        "tel": fake.phone_number(),
        "bank_name": fake.company(),
        "branch_name": fake.city(),
        "account_number": fake.bban(),
        "self_promotion": fake.text(max_nb_chars=500),
    }


def generate_profile(account_id, rank_ids):
    rank_id = random.choice(rank_ids)
    return {
        "display_name": fake.user_name(),
        "rank_id": rank_id,
        "self_promotion": fake.text(max_nb_chars=200),
        "status": "受付中",
        "price": random.randint(1000, 10000),
        "account_id": account_id
    }


def generate_nsfw_options(account_id):
    return {
        "allowed": random.choice([True, False]),
        "price": random.randint(500, 5000),
        "extreme_allowed": random.choice([True, False]),
        "extreme_surcharge": random.randint(1000, 8000),
        "account_id": account_id
    }


def generate_profile_image(account_id):
    mime_types = ['image/jpeg', 'image/png', 'image/gif']
    mime_type = random.choice(mime_types)
    file_extension = mime_type.split('/')[-1]
    filename = f"{account_id}_{fake.word()}_{random.randint(1000, 9999)}.{file_extension}"
    size = random.randint(1024, 2048)
    return {
        "account_id": account_id,
        "mime_type": mime_type,
        "size": size,
        "path": filename
    }


def generate_voice(account_id):
    mime_types = ['audio/mpeg', 'audio/wav', 'audio/ogg']
    mime_type = random.choice(mime_types)
    file_extension = mime_type.split('/')[-1]
    filename = f"{account_id}_{fake.word()}_{random.randint(1000, 9999)}.{file_extension}"
    voice_hash = hashlib.sha256(
        (str(account_id) + filename).encode()).hexdigest()
    return {
        "title": fake.sentence(nb_words=4),
        "account_id": account_id,
        "mime_type": mime_type,
        "path": filename,
        "hash": voice_hash
    }


def get_all_rank_ids():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM actor_ranks")
        rank_ids = [row[0] for row in cursor.fetchall()]  # type: ignore
        cursor.close()
        conn.close()
        return rank_ids
    except mysql.connector.Error as err:
        print(f"Error retrieving rank ids: {err}")
        return []


def insert_account_role(account_id, role_name):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        # Insert role if not exists
        cursor.execute(
            "INSERT IGNORE INTO roles (name) VALUES (%s)", (role_name,))
        # Get role_id
        cursor.execute("SELECT id FROM roles WHERE name = %s", (role_name,))
        role_id_result = cursor.fetchone()
        if role_id_result:
            role_id = role_id_result[0]  # type: ignore
            # Insert into account_role
            cursor.execute(
                "INSERT INTO account_role (account_id, role_id) VALUES (%s, %s)", (account_id, role_id))  # type: ignore
            conn.commit()
        else:
            print(f"Role '{role_name}' not found and could not be inserted.")
        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        print(f"Error inserting data into account_role: {err}")


def generate_tags():
    categories = ["genre", "mood", "theme"]
    tags = []
    for category in categories:
        for _ in range(5):  # 各カテゴリに5つのタグを生成
            tags.append({
                "name": fake.word(),
                "category": category
            })
    return tags


def insert_tags(tags):
    tag_ids = []
    for tag in tags:
        tag_id = insert_data("tags", tag, exclude_columns=["id"])
        if tag_id:
            tag_ids.append(tag_id)
    return tag_ids


def assign_tags_to_voice(voice_id, tag_ids):
    for tag_id in tag_ids:
        insert_data("voice_tag", {"voice_id": voice_id, "tag_id": tag_id})


def generate_and_insert_fake_data(num_accounts=10, num_voices_per_account=5):
    # タグを生成して挿入
    tags = generate_tags()
    tag_ids = insert_tags(tags)

    # Define actor ranks with descriptions
    actor_ranks_data = [
        {"name": "Bronze", "description": "Entry-level actor rank for new users."},
        {"name": "Silver", "description": "Mid-level actor rank with increased visibility."},
        {"name": "Gold", "description": "High-tier actor rank offering premium features."},
        {"name": "Platinum", "description": "Exclusive actor rank with top-tier benefits."},
        {"name": "Diamond", "description": "Elite actor rank for top-performing actors."}
    ]

    # Insert actor ranks into the database
    for rank in actor_ranks_data:
        insert_data("actor_ranks", rank)

    # Retrieve all rank_ids
    rank_ids = get_all_rank_ids()
    if not rank_ids:
        print("No ranks available in actor_ranks table.")
        return

    for _ in range(num_accounts):
        # Generate and insert account
        account_data = generate_account()
        account_id = insert_data(
            "accounts", account_data, exclude_columns=["id"])
        if account_id is None:
            continue

        # Generate and insert profile data
        profile_data = generate_profile(account_id, rank_ids)
        insert_data("actor_profiles", profile_data, exclude_columns=["id"])

        # Generate and insert nsfw_options data
        nsfw_options_data = generate_nsfw_options(account_id)
        insert_data("nsfw_options", nsfw_options_data, exclude_columns=["id"])

        # Generate and insert profile_image data
        profile_image_data = generate_profile_image(account_id)
        insert_data("profile_images", profile_image_data,
                    exclude_columns=["id"])

        # Generate and insert voices data
        for _ in range(num_voices_per_account):
            voice_data = generate_voice(account_id)
            voice_id = insert_data("voices", voice_data,
                                   exclude_columns=["id"])
            if voice_id:
                # Assign tags to voice here if needed
                pass

        # Generate and insert voices data
        for _ in range(num_voices_per_account):
            voice_data = generate_voice(account_id)
            voice_id = insert_data("voices", voice_data,
                                   exclude_columns=["id"])
            if voice_id:
                # タグを声に関連付ける
                assign_tags_to_voice(voice_id, random.sample(
                    tag_ids, k=3))  # 各声に3つのタグをランダムに割り当て

        # Assign role to account
        insert_account_role(account_id, "actor")

    print("Fake data generation and insertion complete.")


if __name__ == "__main__":
    generate_and_insert_fake_data(num_accounts=20, num_voices_per_account=10)
