from json import load
from os import path
from urllib.request import urlopen

from sbw import Monitor

from works import raw_path


def download_raw_dataset(article_id: str,
                         timeout: int = 120):
    """
    Download all files associated with a figshare article.

    :param article_id: identifier of the figshare article.
    :type article_id: int

    :param timeout: timeout in seconds for each download request.
    :type timeout: int
    """
    url = f"https://api.figshare.com/v2/articles/{article_id}"

    with urlopen(url, timeout=timeout) as response:
        information = load(response)

    file_list = information["files"]
    for index, file in enumerate(file_list):
        file_name, url, size = file["name"], file["download_url"], file["size"]

        # skip this file that has already been downloaded.
        file_path = f"{raw_path}{file_name}"
        if path.exists(file_path) and path.getsize(file_path) == size:
            continue

        print(f"Downloading ({index + 1}/{len(file_list)}): {file_name}")

        monitor, downloaded = Monitor(), 0
        with urlopen(url, timeout=timeout) as response:
            with open(file_path, "wb") as output:
                while chunk := response.read(1024 * 1024):
                    downloaded += len(chunk)
                    output.write(chunk)
                    monitor(downloaded, size)

        if path.getsize(file_path) != size:
            raise IOError(f"File size mismatch: {file_name}")


if __name__ == "__main__":
    download_raw_dataset(article_id="34633638")
