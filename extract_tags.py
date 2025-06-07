import matplotlib.pyplot as plt


def extract_all_tags(filename):
    tags = []
    for files in filename.iterdir():
        if files.is_file() and files.suffix == ".txt":
            with open(files, "r", encoding="utf-8") as f:
                tags.extend([line.strip() for line in f if line.strip()])
    return tags


if __name__ == "__main__":
    from pathlib import Path

    # Specify the directory containing the tag files
    tags_directory = Path("./tags")

    # Extract all tags from the files in the directory
    all_tags = extract_all_tags(tags_directory)

    print(f"Total unique tags extracted: {len(set(all_tags))}")

    # Save the unique tags to a file
    with open("character_tags.txt", "w", encoding="utf-8") as f:
        for tag in set(all_tags):
            f.write(f"{tag}\n")
