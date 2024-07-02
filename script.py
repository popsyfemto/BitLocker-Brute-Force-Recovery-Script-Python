import os
import hashlib
import itertools
import subprocess
from concurrent.futures import ThreadPoolExecutor
from numba import cuda
import numpy as np

# Constants
GROUP_SIZE = 6
NUM_GROUPS = 8
KEY_SIZE = GROUP_SIZE * NUM_GROUPS
CHUNK_SIZE = 10**9  # Define a chunk size that Numba can handle

def list_drives():
    drives = []
    for drive in range(65, 91):
        drive_letter = chr(drive)
        if os.path.exists(f"{drive_letter}:/"):
            drives.append(f"{drive_letter}:\\")
    return drives

def select_drive(drives):
    print("Available drives:")
    for i, drive in enumerate(drives):
        print(f"{i}: {drive}")
    drive_index = int(input("Select a drive by index: "))
    return drives[drive_index]

def check_bitlocker_status(drive):
    command = f"Get-BitLockerVolume -MountPoint {drive} | Select-Object -ExpandProperty ProtectionStatus"
    result = subprocess.run(["powershell", "-Command", command], capture_output=True, text=True)
    status = result.stdout.strip()
    return status == "On"

def list_gpus():
    cuda.select_device(0)  # Ensure that Numba CUDA API is initialized
    gpus = cuda.list_devices()
    return gpus

def select_gpu(gpus):
    print("Available GPUs:")
    for i, gpu in enumerate(gpus):
        print(f"{i}: {gpu.name.decode()}")
    gpu_index = int(input("Select a GPU by index: "))
    cuda.select_device(gpu_index)
    return gpu_index

# GPU Kernel for Brute Force
@cuda.jit
def brute_force_kernel(start, end, target_hash, found, result):
    idx = cuda.grid(1)
    if idx >= end - start:
        return

    # Generate the password to try
    password = start + idx

    # Convert password to string and encode it
    password_str = f'{password:048d}'  # Ensure password is 48 digits
    password_bytes = password_str.encode('utf-8')

    # Hash the password
    hash_object = hashlib.sha256(password_bytes)
    hashed_password = hash_object.hexdigest()

    # Check if the hash matches the target hash
    if hashed_password == target_hash:
        found[0] = True
        result[0] = password

def brute_force_chunk(start, end, target_hash):
    threads_per_block = 128
    blocks_per_grid = (end - start + threads_per_block - 1) // threads_per_block

    found = cuda.to_device(np.array([False], dtype=np.bool_))
    result = cuda.to_device(np.array([-1], dtype=np.int64))

    brute_force_kernel[blocks_per_grid, threads_per_block](start, end, target_hash, found, result)

    cuda.synchronize()

    if found.copy_to_host()[0]:
        print(f"Password found: {result.copy_to_host()[0]:048d}")
        return result.copy_to_host()[0]
    else:
        print(f"Password not found in range {start} to {end}")
        return None

def brute_force_bitlocker(target_hash):
    max_password = 10**KEY_SIZE

    with ThreadPoolExecutor() as executor:
        futures = []
        for start in range(0, max_password, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, max_password)
            futures.append(executor.submit(brute_force_chunk, start, end, target_hash))

        for future in futures:
            result = future.result()
            if result is not None:
                print(f"Recovery key found: {result:048d}")
                return result
    print("Recovery key not found")
    return None

def main():
    drives = list_drives()
    selected_drive = select_drive(drives)
    
    if check_bitlocker_status(selected_drive):
        print(f"BitLocker protection detected on drive {selected_drive}")

        gpus = list_gpus()
        selected_gpu = select_gpu(gpus)

        target_hash = hashlib.sha256(b"123456123456123456123456123456123456123456123456").hexdigest()  # Example target hash

        brute_force_bitlocker(target_hash)
    else:
        print(f"No BitLocker protection detected on drive {selected_drive}")

if __name__ == "__main__":
    main()
