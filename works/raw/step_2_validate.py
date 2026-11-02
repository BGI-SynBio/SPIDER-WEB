from hashlib import md5
from os import path, remove

from sbw import Monitor

from works import raw_path

data = {}
with open(f"{raw_path}checklist.txt") as file:
    for line in file.readlines():
        info = line.strip().split(",")
        data[info[0]] = (info[1], int(info[2]))


def obtain_info(file_path: str):
    md5_hash = md5()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    return md5_hash.hexdigest(), path.getsize(file_path)


if __name__ == "__main__":
    monitor = Monitor()

    print("Verify the intermediate outputs from reproducible computational procedures:")
    total, saved_paths, saved_info = len(data), [], []
    for process, (child_path, (expected_md5, expected_size)) in enumerate(data.items()):
        used_path = f"{raw_path}{child_path}"
        if path.exists(used_path):
            obtained_md5, obtained_size = obtain_info(file_path=used_path)
            if obtained_md5 != expected_md5 or obtained_size != expected_size:
                shared_info_1 = f"expected MD5 vs obtained MD5 = {expected_md5} vs {obtained_md5}"
                shared_info_2 = f"expected size vs obtained size = {expected_size} vs {obtained_size}"
                saved_info.append(f"{used_path} mismatch detected, redownload required ({shared_info_1}; {shared_info_2}).")
                saved_paths.append(used_path)
        else:
            saved_info.append(f"{child_path} does not exist.")
        monitor(process + 1, total, extra={"error number": len(saved_info)})
    if len(saved_info):
        for info in saved_info:
            print(info)
        for used_path in saved_paths:
            remove(used_path)
