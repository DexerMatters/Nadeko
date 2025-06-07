import datetime
from pybooru import Danbooru
import os
import asyncio
import aiohttp
import time
from pathlib import Path
import csv

client = Danbooru("danbooru")
# client.api_key = "41WNoVMBZjy3t81aL7KzscFp"


async def download_urls(urls, output_dir="downloads", max_concurrent=5):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    # Semaphore for limiting concurrent downloads
    semaphore = asyncio.Semaphore(max_concurrent)

    async def download_single(url):
        async with semaphore:
            try:
                # Create unique filename from URL
                filename = f"{int(time.time())}_{hash(url) % 10000}.jpg"
                filepath = output_dir / filename

                # Use timeout to avoid hanging downloads
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=30) as response:
                        if response.status == 200:
                            content = await response.read()
                            with open(filepath, "wb") as f:
                                f.write(content)
                            print(f"Downloaded: {url} -> {filepath}")
                            return str(filepath)
                        else:
                            print(f"Failed to download {url}: HTTP {response.status}")
                            exit(1)
                            return None
            except Exception as e:
                print(f"Error downloading {url}: {str(e)}")
                return None

            # Rate limiting to be kind to the server
            await asyncio.sleep(0.5)

    # Create and gather tasks
    tasks = [download_single(url) for url in urls if url]
    results = await asyncio.gather(*tasks)

    # Filter out failed downloads
    return [path for path in results if path]


def download_images(urls, output_dir="downloads", max_concurrent=5):
    """Synchronous wrapper for the async download function"""
    if not urls:
        print("No URLs to download")
        return []

    return asyncio.run(download_urls(urls, output_dir, max_concurrent))


def should_filter_post(post):
    """Check if a post should be filtered out based on tag requirements"""
    tag_string = post.get("tag_string", "").lower()
    rating = post.get("rating", "").lower()

    unwanted_ratings = ["q", "e", "x"]
    if rating in unwanted_ratings:
        return True

    # Must contain "solo"
    if "solo" not in tag_string:
        return True

    # Must not contain greyscale, monochrome, or spot_color
    unwanted_tags = ["greyscale", "monochrome", "spot_color"]
    if any(unwanted_tag in tag_string for unwanted_tag in unwanted_tags):
        return True

    # Must one character tag
    tag_count_character = post.get("tag_count_character", 0)
    if tag_count_character != 1:
        return True

    return False


# Example usage
if __name__ == "__main__":
    tags = []
    start = False
    # Read tags from tags_filtered.txt, one tag per line
    with open("character_tags.txt", "r", encoding="utf-8") as f:
        for line in f:
            tag = line.strip()
            if tag:
                tags.append(tag)

    expected_sample_count = 40
    min_required = 30
    nth = 0
    for tag in tags:
        nth += 1
        print(f"Processing tag {nth}/{len(tags)}: {tag}")
        # if tag == "mohammed_avdol":
        #     start = True
        # if not start:
        #     continue

        image_count = len(os.listdir(Path("data") / tag))
        compensation = expected_sample_count - image_count
        if image_count == 0:
            print(f"Tag '{tag}' is not available, skipping.")
            continue
        if compensation > 0:
            print(
                f"Tag '{tag}' already has {image_count} images, need {compensation} more."
            )
            sample_count = compensation
        else:
            print(f"Tag '{tag}' already has enough images ({image_count}), skipping.")
            continue

        print(f"Processing tag: {tag} ({sample_count} images)")
        urls = []
        # Try up to 4 attempts with different random/limit settings if error or not enough images
        attempts = [
            {"limit": sample_count * 4, "random": True},
            {"limit": sample_count * 3, "random": True},
            {"limit": sample_count * 2, "random": True},
            {"limit": sample_count * 1, "random": True},
        ]
        for attempt in attempts:
            try:
                posts = client.post_list(
                    tags=tag,
                    limit=attempt["limit"],
                    random=attempt["random"],
                )
                for post in posts:
                    # Apply filtering based on tag_string
                    if should_filter_post(post):
                        continue

                    min_url = None
                    if "preview_file_url" in post:
                        min_url = post["preview_file_url"]
                    elif "file_url" in post:
                        min_url = post["file_url"]
                    elif "media_asset" in post and "variants" in post["media_asset"]:
                        for variant in post["media_asset"]["variants"]:
                            if "url" in variant:
                                min_url = variant["url"]
                                break
                    if min_url:
                        urls.append(min_url)
                    if len(urls) >= sample_count:
                        break
                if len(urls) >= min_required:
                    break  # Got enough images, stop retrying
            except Exception as e:
                print(
                    f"Error fetching posts for tag '{tag}' (attempt {attempts.index(attempt)+1}): {e}"
                )
                if attempts.index(attempt) == len(attempts) - 1:
                    print(f"Failed to fetch enough images for tag '{tag}', interrupt.")
                    continue
                continue
        if len(urls) < min_required:
            print(
                f"!!!Warning: Only found {len(urls)} images for tag '{tag}' (minimum required: {min_required})"
            )
        tag_dir = Path("data") / tag
        tag_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {min(len(urls), sample_count)} images for tag '{tag}'...")
        if download_images(urls[:sample_count], output_dir=tag_dir) is None:
            print(f"Failed to download images for tag '{tag}'")
            exit(1)
    print("All downloads complete.")
