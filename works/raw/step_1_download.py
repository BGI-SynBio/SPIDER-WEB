from json import load
from os import path
from urllib.request import urlopen

from sbw import Monitor

from works import raw_path


def obtain_file_list(article_id: int) \
        -> list:
    """
    Obtain the list of files associated with a figshare article.

    :param article_id: identifier of the figshare article.
    :type article_id: int

    :return: list of file metadata dictionaries.
    :rtype: list
    """
    url = f"https://api.figshare.com/v2/articles/{article_id}"

    with urlopen(url, timeout=30) as response:
        information = load(response)

    return information["files"]


def download_file(file: dict,
                  process: int,
                  total: int,
                  timeout: int = 120):
    """
    Download a file from figshare and verify its size.

    :param file: dictionary containing file metadata.
    :type file: dict

    :param process: index of the current file in the download sequence.
    :type process: int

    :param total: total number of files to download.
    :type total: int

    :param timeout: timeout in seconds for the download request.
    :type timeout: int
    """
    file_name, url, size = file["name"], file["download_url"], file["size"]

    # skip this file that has already been downloaded.
    file_path = f"{raw_path}{file_name}"
    if path.exists(file_path) and path.getsize(file_path) == size:
        return

    print(f"Downloading ({process}/{total}): {file_name}")
    monitor, downloaded = Monitor(), 0
    with urlopen(url, timeout=timeout) as response:
        with open(file_path, "wb") as output:
            while chunk := response.read(1024 * 1024):
                downloaded += len(chunk)
                output.write(chunk)
                monitor(downloaded, size)

    if path.getsize(file_path) != size:
        raise IOError(f"File size mismatch: {file_name}")


def download_raw_dataset(article_id: int,
                         timeout: int = 120):
    """
    Download all files associated with a figshare article.

    :param article_id: identifier of the figshare article.
    :type article_id: int

    :param timeout: timeout in seconds for each download request.
    :type timeout: int
    """
    file_list = obtain_file_list(article_id)
    for index, file in enumerate(file_list):
        download_file(file, index + 1, len(file_list), timeout)


if __name__ == "__main__":
    download_raw_dataset(article_id=34633638)
