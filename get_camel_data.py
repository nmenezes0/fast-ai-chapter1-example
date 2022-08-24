"""Script to get camel images for experimentation."""

from pathlib import Path
from time import sleep

from duckduckgo_search import ddg_images
from fastai.vision.all import download_images, resize_images
from fastcore.all import L

PARENT_PATH = Path(__file__).parent
CAMELS_DATA_PATH = Path(PARENT_PATH, "data/camels")
CATEGORIES = ["bactrian", "dromedary"]


def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return L(ddg_images(term, max_results=max_images)).itemgot("image")


def get_dataset():
    for category in CATEGORIES:
        dest = Path(CAMELS_DATA_PATH, category)
        download_images(dest, urls=search_images(f"{category} camel photo"))
        sleep(10)  # Pause between searches to avoid over-loading server
        download_images(dest, urls=search_images(f"{category} camel sun photo"))
        sleep(10)
        download_images(dest, urls=search_images(f"{category} camel shade photo"))
        sleep(10)
        resize_images(dest, max_size=400, dest=dest)


if __name__ == "__main__":
    get_dataset()
