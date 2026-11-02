from hashlib import md5
from os import path, remove

from sbw import Monitor

from works import exp_path

data_1 = {}
with open(f"{exp_path}checklist-1.txt") as file:
    for line in file.readlines():
        info = line.strip().split(",")
        data_1[info[0]] = (info[1], int(info[2]))


data_2 = {}
with open(f"{exp_path}checklist-2.txt") as file:
    for line in file.readlines():
        info = line.strip().split(",")
        data_2[info[0]] = (info[1], int(info[2]))


def obtain_info(file_path: str):
    md5_hash = md5()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    return md5_hash.hexdigest(), path.getsize(file_path)


if __name__ == "__main__":
    monitor = Monitor()

    print("Verify the raw data from the error-tolerant data retrieval experiment:")
    total, saved_paths, saved_info_1 = len(data_1), [], []
    for process, (child_path, (expected_md5, expected_size)) in enumerate(data_1.items()):
        used_path = f"{exp_path}1/{child_path}"
        if path.exists(used_path):
            obtained_md5, obtained_size = obtain_info(file_path=used_path)
            if obtained_md5 != expected_md5 or obtained_size != expected_size:
                shared_info_1 = f"expected MD5 vs obtained MD5 = {expected_md5} vs {obtained_md5}"
                shared_info_2 = f"expected size vs obtained size = {expected_size} vs {obtained_size}"
                saved_info_1.append(f"{used_path} mismatch detected, redownload required ({shared_info_1}; {shared_info_2}).")
                saved_paths.append(used_path)
        else:
            saved_info_1.append(f"{child_path} does not exist.")
        monitor(process + 1, total, extra={"error number": len(saved_info_1)})
    if len(saved_info_1):
        for info in saved_info_1:
            print(info)
        for used_path in saved_paths:
            remove(used_path)

    print()

    print("Verify the raw data from the real-time data retrieval experiment:")
    total, saved_paths, saved_info_2 = len(data_2), [], []
    for process, (child_path, (expected_md5, expected_size)) in enumerate(data_2.items()):
        used_path = f"{exp_path}2/{child_path}"
        if path.exists(used_path):
            obtained_md5, obtained_size = obtain_info(file_path=used_path)
            if obtained_md5 != expected_md5 or obtained_size != expected_size:
                shared_info_1 = f"expected MD5 vs obtained MD5 = {expected_md5} vs {obtained_md5}"
                shared_info_2 = f"expected size vs obtained size = {expected_size} vs {obtained_size}"
                saved_info_2.append(f"{used_path} mismatch detected, redownload required ({shared_info_1}; {shared_info_2}).")
                saved_paths.append(used_path)
        else:
            saved_info_2.append(f"{child_path} does not exist.")
        monitor(process + 1, total, extra={"error number": len(saved_info_2)})
    if len(saved_info_2):
        for info in saved_info_2:
            print(info)
        for used_path in saved_paths:
            remove(used_path)

    if len(saved_info_1) or len(saved_info_2):
        raise SystemExit(1)
