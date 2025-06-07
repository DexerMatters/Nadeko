from pybooru import Danbooru
import base64

client = Danbooru("danbooru")


if __name__ == "__main__":
    with open("anime_tags.txt", "r", encoding="utf-8") as f:
        tags = [line.strip() for line in f if line.strip()]

    for tag in tags:
        related = client.tag_related(tag, "4", order="Cosine")["related_tags"]
        count = 0
        # new file
        with open(
            f"./tags/{base64.urlsafe_b64encode(bytes(tag, encoding='utf-8'))}.txt",
            "w",
            encoding="utf-8",
        ) as f:
            for related_tag in related:
                if related_tag["overlap_coefficient"] < 0.9:
                    continue
                related_tag = related_tag["tag"]
                if related_tag["post_count"] < 500:
                    continue
                count += 1
                f.write(f"{related_tag['name']}\n")
            if count == 0:
                count += 1
                try:
                    f.write(f"{related[0]['tag']['name']}\n")
                except IndexError:
                    print(f"No related tags found for {tag}")
                    continue
        print(f"Tag: {tag}, Collected tags: {count}")
