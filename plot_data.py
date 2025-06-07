import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns
from pathlib import Path


def count_images_per_tag(data_dir):
    """Count the number of images for each tag in the data directory."""
    tag_counts = {}

    # Iterate through tag directories
    for tag in os.listdir(data_dir):
        tag_path = os.path.join(data_dir, tag)

        # Skip if not a directory
        if not os.path.isdir(tag_path):
            continue

        # Count image files in the tag directory
        image_count = sum(
            1
            for file in os.listdir(tag_path)
            if os.path.isfile(os.path.join(tag_path, file))
            and file.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
        )

        tag_counts[tag] = image_count

    return tag_counts


def plot_tag_distribution(tag_counts, output_dir):
    """Create multiple plots showing the distribution of images across tags."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Sort tags by count (descending)
    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    tags, counts = zip(*sorted_tags)

    # 1. Plot Top 30 tags
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")

    # Use only top 30 tags for the bar plot
    top_n = 30
    top_tags = tags[:top_n]
    top_counts = counts[:top_n]

    bars = plt.bar(
        range(len(top_tags)),
        top_counts,
        color=sns.color_palette("viridis", len(top_tags)),
    )

    plt.xlabel("Tags", fontsize=12)
    plt.ylabel("Number of Images", fontsize=12)
    plt.title(f"Top {top_n} Tags by Image Count", fontsize=14)

    plt.xticks(range(len(top_tags)), top_tags, rotation=45, ha="right")

    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{height}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    output_path = os.path.join(output_dir, "top_tags_distribution.png")
    plt.savefig(output_path, dpi=300)
    print(f"Top tags plot saved to {output_path}")
    plt.close()

    # 2. Histogram of image counts per tag
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    plt.hist(counts, bins=50, color="skyblue", edgecolor="black")
    plt.xlabel("Number of Images", fontsize=12)
    plt.ylabel("Number of Tags", fontsize=12)
    plt.title("Distribution of Image Counts per Tag", fontsize=14)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "tag_count_histogram.png")
    plt.savefig(output_path, dpi=300)
    print(f"Histogram saved to {output_path}")
    plt.close()

    # 3. CDF plot for image distribution
    plt.figure(figsize=(12, 8))

    # Calculate the cumulative distribution
    sorted_counts = sorted(counts)
    cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)

    plt.plot(sorted_counts, cumulative, "b-", linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Number of Images per Tag", fontsize=12)
    plt.ylabel("Cumulative Proportion of Tags", fontsize=12)
    plt.title("Cumulative Distribution of Image Counts", fontsize=14)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "cumulative_distribution.png")
    plt.savefig(output_path, dpi=300)
    print(f"CDF plot saved to {output_path}")
    plt.close()

    # 4. Tag count distribution by frequency ranges
    ranges = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    labels = [f"{ranges[i]}-{ranges[i+1]-1}" for i in range(len(ranges) - 1)]
    labels.append("40+")

    # Count tags in each range
    range_counts = [0] * len(labels)
    for count in counts:
        placed = False
        for i in range(len(ranges) - 1):
            if ranges[i] <= count < ranges[i + 1]:
                range_counts[i] += 1
                placed = True
                break
        if not placed and count >= 40:
            range_counts[-1] += 1

    plt.figure(figsize=(12, 8))
    bars = plt.bar(labels, range_counts, color=sns.color_palette("muted", len(labels)))

    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{height}",
            ha="center",
            va="bottom",
        )

    plt.xlabel("Number of Images Range", fontsize=12)
    plt.ylabel("Number of Tags", fontsize=12)
    plt.title("Distribution of Tags by Image Count Range", fontsize=14)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "tag_range_distribution.png")
    plt.savefig(output_path, dpi=300)
    print(f"Range distribution plot saved to {output_path}")
    plt.close()

    # 5. Plot images indexed by tag (tag as numbers)
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")

    # Create a numeric index for each tag
    tag_indices = np.arange(len(tags))

    # Print diagnostic information
    print(f"\nPlot diagnostic info:")
    print(f"Number of tags to plot: {len(tag_indices)}")
    print(f"Max count: {max(counts)}, Min count: {min(counts)}")

    # Create the filled area plot
    plt.fill_between(tag_indices, counts, color="steelblue", alpha=0.7)

    # Add a line on top of the filled area for clarity
    plt.plot(tag_indices, counts, color="navy", linewidth=1.0)

    # Add some marker points for better visibility
    marker_interval = max(1, len(tag_indices) // 40)
    plt.scatter(
        tag_indices[::marker_interval],
        [counts[i] for i in range(0, len(counts), marker_interval)],
        color="red",
        s=15,
        zorder=3,
    )

    # Use logarithmic scale if there's high variance in counts
    if max(counts) / (min(counts) + 0.1) > 20:  # Add 0.1 to avoid division by zero
        plt.yscale("log")
        plt.ylabel("Number of Images (log scale)", fontsize=12)
    else:
        plt.ylabel("Number of Images", fontsize=12)

    plt.xlabel("Tag Index", fontsize=12)
    plt.title("Image Distribution by Tag Index", fontsize=14)

    # Add grid lines for better readability
    plt.grid(True, alpha=0.3)

    # Add ticks at regular intervals if there are many tags
    if len(tag_indices) > 30:
        tick_interval = max(1, len(tag_indices) // 20)
        plt.xticks(tag_indices[::tick_interval])
    else:
        plt.xticks(tag_indices)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "tag_index_distribution.png")
    plt.savefig(output_path, dpi=300)
    print(f"Tag index plot saved to {output_path}")
    plt.close()

    # 6. Print statistics summary
    print("\nTag Statistics:")
    print(f"Total tags: {len(tag_counts)}")
    print(f"Total images: {sum(counts)}")
    print(f"Maximum images for a tag: {max(counts)} (Tag: {tags[0]})")
    print(f"Minimum images for a tag: {min(counts)}")
    print(f"Average images per tag: {sum(counts) / len(counts):.2f}")
    print(f"Median images per tag: {sorted_counts[len(sorted_counts) // 2]}")


if __name__ == "__main__":
    # Define input and output directories
    data_dir = "./data"
    output_dir = "./data_plots"

    # Count images per tag
    tag_counts = count_images_per_tag(data_dir)

    # Create and save the plot
    plot_tag_distribution(tag_counts, output_dir)

    # The statistics are now printed in the plot_tag_distribution function
