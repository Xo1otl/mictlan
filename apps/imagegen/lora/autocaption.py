import ollama
import os
import json
from concurrent.futures import ThreadPoolExecutor

MODEL = 'llama3.2-vision'
IMAGE_DIR = './pikazo_images'
OUTPUT_FILE = 'image_captions.json'  # File to store the captions in JSON format
SYSTEM_PROMPT = """\
You are an assistant who describes the content and composition of images.
Describe only what you see in the image, not what you think the image is about.
Be factual and literal. Do not use metaphors or similes. Be concise.
"""
PROMPT = """\
Please describe this image in 30 to 40 words.
"""


def get_caption(image_path):
    try:
        res = ollama.chat(
            model=MODEL,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': PROMPT, 'images': [image_path]}]
        )
        caption = res['message']['content']
        # Print to console
        print(f"File: {os.path.basename(image_path)}, Caption: {caption}")
        return {'file': os.path.basename(image_path), 'caption': caption}
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def process_images(directory):
    image_paths = [os.path.join(directory, file) for file in os.listdir(directory)
                   if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    captions = []
    # Process up to 4 images concurrently
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(get_caption, image_paths)
        for result in results:
            if result:
                captions.append(result)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(captions, f, indent=4)


process_images(IMAGE_DIR)
