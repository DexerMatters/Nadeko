import os

DATA_DIR = "./data"
MAX_IMAGES = 40
MIN_IMAGES = 25

for tag in os.listdir(DATA_DIR):
    tag_path = os.path.join(DATA_DIR, tag)
    if not os.path.isdir(tag_path):
        continue
    images = [
        f for f in os.listdir(tag_path) if os.path.isfile(os.path.join(tag_path, f))
    ]
    if len(images) > MAX_IMAGES:
        # Sort images for deterministic truncation (optional)
        images.sort()
        surplus = images[MAX_IMAGES:]
        for img in surplus:
            os.remove(os.path.join(tag_path, img))
    elif len(images) < MIN_IMAGES:
        print(f"deleting tag '{tag}' due to insufficient images ({len(images)})")
        # Remove the tag directory and its contents
        import shutil

        shutil.rmtree(tag_path)
