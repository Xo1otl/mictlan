import mysql.connector
from faker import Faker
import random
import hashlib
from infra import koemade
import os

fake = Faker('ja_JP')

# MySQL Database Configuration
db_config = {
    "host": "mysql",
    "user": koemade.MYSQL_USER,
    "password": koemade.MYSQL_PASSWORD,
    "database": koemade.MYSQL_DB
}


# Helper function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# Function to generate fake signup request data
def generate_signup_request():
    return {
        "name": fake.name(),
        "furigana": fake.kana_name(),
        "address": fake.address(),
        "email": fake.email(),
        "tel": fake.phone_number(),
        "bank_name": fake.company(),
        "branch_name": fake.city(),
        "account_number": fake.bban(),
        "self_promotion": fake.text(max_nb_chars=500)
    }


# Function to generate fake account data
def generate_account():
    username = fake.unique.user_name() + "@example.com"  # Ensure unique usernames
    password = hash_password("password123")  # Default password, hashed
    return {
        "username": username,
        "password": password
    }


# Function to generate fake profile data
def generate_profile(account_id):
    categories = ["アニメ", "ゲーム", "ナレーション", "CM", "教育", "その他"]
    return {
        "display_name": fake.user_name(),
        "category": random.choice(categories),
        "self_promotion": fake.text(max_nb_chars=200),
        "price": random.randint(1000, 10000),
        "account_id": account_id
    }


# Function to generate fake actor data
def generate_actor_r(account_id):
    return {
        "ok": random.choice([True, False]),
        "price": random.randint(500, 5000),
        "hard_ok": random.choice([True, False]),
        "hard_surcharge": random.randint(1000, 8000),
        "account_id": account_id
    }


# Function to generate fake profile image data
def generate_profile_image(account_id, image_dir="profile_images"):
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    mime_types = ['image/jpeg', 'image/png', 'image/gif']
    mime_type = random.choice(mime_types)
    file_extension = mime_type.split('/')[-1]
    filename = f"{account_id}_{fake.word()}_{random.randint(1000, 9999)}.{file_extension}"
    filepath = os.path.join(image_dir, filename)

    # Create dummy image file
    with open(filepath, 'wb') as f:
        f.write(os.urandom(random.randint(1024, 2048)))

    return {
        "account_id": account_id,
        "mime_type": mime_type,
        "size": os.path.getsize(filepath),
        "path": filepath
    }


# Function to generate fake voice data
def generate_voice(account_id, voice_dir="voices"):
    if not os.path.exists(voice_dir):
        os.makedirs(voice_dir)

    mime_types = ['audio/mpeg', 'audio/wav', 'audio/ogg']
    mime_type = random.choice(mime_types)
    file_extension = mime_type.split('/')[-1]
    filename = f"{account_id}_{fake.word()}_{random.randint(1000, 9999)}.{file_extension}"
    filepath = os.path.join(voice_dir, filename)

    with open(filepath, 'wb') as f:
        f.write(os.urandom(random.randint(1024, 2048)))

    return {
        "title": fake.sentence(nb_words=4),
        "account_id": account_id,
        "mime_type": mime_type,
        "filename": filename,
    }


# Function to insert data into a table
def insert_data(table_name, data):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        placeholders = ', '.join(['%s'] * len(data))
        columns = ', '.join(data.keys())
        sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        cursor.execute(sql, list(data.values()))
        conn.commit()
        insert_id = cursor.lastrowid
        cursor.close()
        conn.close()
        return insert_id
    except mysql.connector.Error as err:
        print(f"Error inserting data into {table_name}: {err}")
        return None


# Function to insert data into account_roles table
def insert_account_role(account_id, role_name):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Get role ID
        cursor.execute(
            "SELECT id FROM roles WHERE role_name = %s", (role_name,))
        role_id_result = cursor.fetchone()

        if role_id_result:
            role_id = role_id_result[0]  # type: ignore
            # Insert into account_roles
            cursor.execute(
                "INSERT INTO account_roles (account_id, role_id) VALUES (%s, %s)", (account_id, role_id))  # type: ignore
            conn.commit()
        else:
            print(f"Role '{role_name}' not found.")

        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        print(f"Error inserting data into account_roles: {err}")


# Function to randomly assign tags to a voice
def assign_voice_tags(voice_id):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        # Get all tag IDs
        cursor.execute("SELECT id FROM voice_tags")
        tag_ids = [row[0] for row in cursor.fetchall()]  # type: ignore

        # Assign 1-3 random tags
        num_tags_to_assign = random.randint(1, min(3, len(tag_ids)))
        assigned_tag_ids = random.sample(tag_ids, num_tags_to_assign)

        # Insert into voice_tag_map
        for tag_id in assigned_tag_ids:
            cursor.execute(
                "INSERT INTO voice_tag_map (voice_id, tag_id) VALUES (%s, %s)", (voice_id, tag_id))  # type: ignore
        conn.commit()
        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        print(f"Error assigning tags to voice: {err}")


# Function to generate and insert all fake data
def generate_and_insert_fake_data(num_accounts=10, num_voices_per_account=5):
    for _ in range(num_accounts):
        # Generate and insert account data
        account_data = generate_account()
        account_id = insert_data("accounts", account_data)
        if account_id:
            insert_account_role(account_id, "actor")

            # Generate and insert signup request data
            signup_request_data = generate_signup_request()
            insert_data("signup_requests", signup_request_data)

            # Generate and insert profile data
            profile_data = generate_profile(account_id)
            insert_data("profiles", profile_data)

            # Generate and insert actor data
            actor_r_data = generate_actor_r(account_id)
            insert_data("actors_r", actor_r_data)

            # Generate and insert profile image data
            profile_image_data = generate_profile_image(account_id)
            insert_data("profile_images", profile_image_data)

            # Generate and insert voices data
            for _ in range(num_voices_per_account):
                voice_data = generate_voice(account_id)
                voice_id = insert_data("voices", voice_data)
                if voice_id:
                    assign_voice_tags(voice_id)

    print("Fake data generation and insertion complete.")


# Main execution
if __name__ == "__main__":
    generate_and_insert_fake_data(num_accounts=20, num_voices_per_account=10)
