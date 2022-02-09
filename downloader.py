import os
import io
import os
import time
import json
import click
import requests
import pandas as pd
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
from typing import Any, Dict, List, Tuple
from toolz.curried import compose, pipe, curry


def _image_worker(args):
    r"""Helper method for parallelizing image downloads."""
    
    image_url, downloader_fn = args
    download_status = downloader_fn(image_url)

    # Sleep for 2 seconds for Imgur, and 0.1 seconds for Reddit and Flickr.
    # This takes care of all request rate limits.
    # Need to test the validity of this heuristic for FashionMnist
    if "imgur" in image_url:
        time.sleep(2.0)
    else:
        time.sleep(0.1)

    return download_status


def annotations_file_reader(filepath, nitems=1000):
    r"""
    This function should read the user supplied annotations file 
    and output the annotations as a list of dictionaries.
    """
    return pipe(filepath, 
                lambda f: pd.read_csv(f), 
                lambda df: [json.loads(row[1].item()) for row in df.iterrows()], 
                lambda l: l[:nitems])


@curry
def image_downloader(url: str, save_to: str, longer_resize: int = 512, time_out: int = 5) -> bool:
    r"""
    Download image from ``url`` and save it to ``save_to``.

    Args:
        url: Image URL to download from.
        save_to: Local path to save the downloaded image.
        time_out: to allow the http request to time out

    Returns:
        Boolean variable indicating whether the download was successful
        (``True``) or not (``False``).
    """
    try:
        # 'response.content' will have our image (as bytes) if successful.
        response = requests.get(url, timeout=time_out)

        # Check if image was downloaded (response must be 200). One exception:
        # Imgur gives response 200 with "removed.png" image if not found.
        # urllib.request.urlretrieve can also be used
        if response.status_code != 200 or "removed.png" in response.url:
            return False

        # Write image to disk if it was downloaded successfully.
        pil_image = Image.open(io.BytesIO(response.content)).convert("RGB")

        # Resize image to longest max size while preserving aspect ratio if
        # longest max size is provided (not -1), and image is bigger.
        if longer_resize > 0:
            image_width, image_height = pil_image.size

            scale = longer_resize / float(max(image_width, image_height))

            if scale != 1.0:
                new_width, new_height = tuple(
                    int(round(d * scale)) for d in (image_width, image_height)
                )
                pil_image = pil_image.resize((new_width, new_height))

        # Save the downloaded image to disk.
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        pil_image.save(save_to)

        return True

    except Exception as err:
        return False   


@click.command()
@click.option(
    "-a", "--annotations", "annotations_filepath", type=click.Path(exists=True),
    help="Path to annotations for downloading images.",
)
@click.option(
    "-o", "--folder", type=click.Path(), default="./datasets/catalog",
    help="""Path to a directory to save images. Images will be saved in sub-
    directories - a different one per subreddit.""",
)
# @click.option(
#     "-r", "--resize", type=int, default=512,
#     help="""Resize longer edge of image, preserving aspect ratio. Set to -1 to
#     prevent resizing.""",
# )
# @click.option(
#     "-j", "--workers", type=int, default=4,
#     help="Number of workers to download images in parallel.",
# )
# @click.option(
#     "-t", "--time_out", type=int, default=5,
#     help="""Limit at which to time out HTTP requests.""",
# )
# @click.option(
#     "-n", "--n_items", type=int, default=1000,
#     help="""Number of items in the annotations list to be downloaded.""",
# )
def download_imgs(
    annotations: str,
    resize: int,
    workers: int,
    folder: str,
    time_out: int,
    n_items: int
):

    annotations_list = annotations_file_reader(annotations, nitems=n_items)

    # Parallelize image downloads.
    with mp.Pool(processes=workers) as p:

        worker_args: List[Tuple] = []
            
        for ann in annotations_list:
            for ix, image_url in enumerate(ann["image_links"]):

                if ann.get("id", None) is not None:
                    image_id = ann["id"]
                    save_to = os.path.join(folder, f"{image_id}", f"{ix}.png")
                    worker_args.append((image_url, 
                                        image_downloader(save_to=save_to, longer_resize=resize, time_out=time_out)))

        download_status = []
        with tqdm(total=len(worker_args), desc="Downloading Images") as pbar:
            for _status in p.imap(_image_worker, worker_args):
                download_status.append(_status)
                pbar.update()

    # How many images were downloaded?
    num_downloaded = sum(download_status)
    print(f"Downloaded {num_downloaded}/{len(worker_args)} images")


if __name__ == "__main__":
    
    # df = pd.read_csv("datasets/catalog/result_10000.csv")
    # annotations_list = [json.loads(row[1].item()) for row in df.iterrows()] 
    annotations = "datasets/catalog/result_10000.csv"
    folder = os.path.join("datasets", "catalog")
    download_imgs(annotations, resize=512, workers=6, folder=folder, time_out=10, n_items=10)