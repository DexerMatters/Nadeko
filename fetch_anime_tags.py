from pybooru import Danbooru

client = Danbooru("danbooru")

if __name__ == "__main__":

    tags = client.tag_list(category="3", order="count", limit=500, page=2)
    results = []
    for tag in tags:
        results.append(tag["name"])

    # Save the tags to a file
    with open("tags.txt", "w", encoding="utf-8") as f:
        for tag in results:
            f.write(f"{tag}\n")
