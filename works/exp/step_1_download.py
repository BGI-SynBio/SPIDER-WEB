from gzip import open as gzip_open
from io import TextIOWrapper
from json import load
from os import path, makedirs, listdir
from shutil import move
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile

from sbw import Monitor

from works import exp_path


def download_exp_dataset(project_id: str,
                         article_id: str,
                         timeout: int = 120):
    """
    Download all files associated with a figshare article.

    :param project_id: identifier of the CNSA project.
    :type project_id: str

    :param article_id: identifier of the figshare article.
    :type article_id: int

    :param timeout: timeout in seconds for each download request.
    :type timeout: int
    """
    for folder_name in ["250F501329017", "250F501329018", "250F501329019", "250F50132901A", "250F50132901B"]:
        makedirs(f"{exp_path}2/{folder_name}/fastq", exist_ok=True)
        makedirs(f"{exp_path}2/{folder_name}/output", exist_ok=True)

    print("Download data in CNGBdb:")
    url_1 = f"https://ftp.cngb.org/pub/CNSA/data7/public_info/{project_id}/data_download_links_{project_id}_ftp.txt"
    collection_1, collection_2 = [], []
    with urlopen(url_1, timeout=timeout) as response:
        for record in TextIOWrapper(response, encoding="utf-8-sig"):
            file_type, file_url = record.strip().split(" ")
            if file_url.endswith(".fq.gz"):
                collection_1.append(file_url)
            else:
                collection_2.append(file_url)

    for index, file_url in enumerate(collection_1):
        file_name = file_url.split("/")[-1]
        if not path.exists(f"{exp_path}1/{file_name}"):
            print(f"Downloading ({index + 1}/{len(collection_1)}): {file_name}")
            monitor, downloaded = Monitor(), 0
            with urlopen(file_url, timeout=timeout) as response:
                file_size = int(response.headers.get("Content-Length"))
                with open(f"{exp_path}1/{file_name}", "wb") as output:
                    while chunk := response.read(1024 * 1024):
                        downloaded += len(chunk)
                        output.write(chunk)
                        # noinspection bad-argument-type
                        monitor(downloaded, file_size)
    for index, file_url in enumerate(collection_2):
        file_name = file_url.split("/")[-1]
        if file_name.endswith(".fastq.gz"):
            used_folder_name, used_file_name = file_name.split("_", 1)
            output_path = f"{exp_path}2/{used_folder_name}/fastq/{used_file_name[:-3]}"
            if not path.exists(output_path):
                print(f"Downloading ({index + 1}/{len(collection_2)}): {file_name}")
                with TemporaryDirectory() as temporary:
                    file_path = path.join(temporary, file_name)
                    monitor, downloaded = Monitor(), 0
                    with urlopen(file_url, timeout=timeout) as response:
                        file_size = int(response.headers.get("Content-Length"))
                        with open(file_path, "wb") as output:
                            while chunk := response.read(1024 * 1024):
                                downloaded += len(chunk)
                                output.write(chunk)
                                monitor(downloaded, file_size)
                        with gzip_open(file_path) as archive:
                            with open(output_path, "wb") as output:
                                while chunk := archive.read(1024 * 1024):
                                    output.write(chunk)
        elif file_name.endswith(".fa.gz"):
            output_path = f"{exp_path}2/all_refs.fa"
            if not path.exists(output_path):
                print(f"Downloading ({index + 1}/{len(collection_2)}): {file_name}")
                with TemporaryDirectory() as temporary:
                    file_path = path.join(temporary, file_name)
                    monitor, downloaded = Monitor(), 0
                    with urlopen(file_url, timeout=timeout) as response:
                        file_size = int(response.headers.get("Content-Length"))
                        with open(file_path, "wb") as output:
                            while chunk := response.read(1024 * 1024):
                                downloaded += len(chunk)
                                output.write(chunk)
                                monitor(downloaded, file_size)
                        with gzip_open(file_path) as archive:
                            with open(output_path, "wb") as output:
                                while chunk := archive.read(1024 * 1024):
                                    output.write(chunk)
        else:
            output_path = f"{exp_path}2/select.align.sort.bam"
            if not path.exists(output_path):
                print(f"Downloading ({index + 1}/{len(collection_2)}): {file_name}")
                monitor, downloaded = Monitor(), 0
                with urlopen(file_url, timeout=timeout) as response:
                    file_size = int(response.headers.get("Content-Length"))
                    with open(output_path, "wb") as output:
                        while chunk := response.read(1024 * 1024):
                            downloaded += len(chunk)
                            output.write(chunk)
                            monitor(downloaded, file_size)

    print("\n")
    print("Download data in figshare:")
    url_2 = f"https://api.figshare.com/v2/articles/{article_id}"
    with urlopen(url_2, timeout=timeout) as response:
        information = load(response)

    file_list = information["files"]
    for index, file in enumerate(file_list):
        zip_file_name, file_url, size = file["name"], file["download_url"], file["size"]
        with TemporaryDirectory() as temporary:
            file_path = path.join(temporary, zip_file_name)
            print(f"Downloading ({index + 1}/{len(file_list)}): {zip_file_name}")
            monitor, downloaded = Monitor(), 0
            with urlopen(file_url, timeout=timeout) as response:
                with open(file_path, "wb") as output:
                    while chunk := response.read(1024 * 1024):
                        downloaded += len(chunk)
                        output.write(chunk)
                        monitor(downloaded, size)

            if path.getsize(file_path) != size:
                raise IOError(f"File size mismatch: {zip_file_name}")

            with ZipFile(file_path) as archive:
                corrupted = archive.testzip()
                if corrupted is not None:
                    raise ValueError(f"Corrupted ZIP member: {corrupted}")
                archive.extractall(temporary)

            folder_name = path.splitext(zip_file_name)[0]
            # noinspection bad-argument-type
            for file_name in listdir(path.join(temporary, folder_name)):
                if file_name.endswith(".npy"):
                    # noinspection bad-argument-type
                    move(path.join(temporary, folder_name, file_name),
                         f"{exp_path}2/{folder_name}/output/{file_name}")
                elif file_name.endswith(".csv"):
                    # noinspection bad-argument-type
                    move(path.join(temporary, folder_name, file_name),
                         f"{exp_path}2/{folder_name}/{file_name}")


if __name__ == "__main__":
    download_exp_dataset(project_id="CNP0010365", article_id="34789395")
