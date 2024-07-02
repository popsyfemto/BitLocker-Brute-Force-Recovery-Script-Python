# BitLocker Brute Force Recovery Script

This project provides a framework for attempting to brute force a BitLocker-protected drive using GPU acceleration. The script checks if a drive is encrypted with BitLocker, allows the user to select a drive and a GPU, and then attempts to brute force the recovery key.

> **Note:** This script is for educational purposes only. Unauthorized access to encrypted data is illegal and unethical. Ensure you have the right to access the data.

## Features

- Lists available drives and checks for BitLocker encryption.
- Lists available GPUs and allows the user to select one.
- Uses CUDA for GPU acceleration.
- Divides the brute force task into manageable chunks for efficient processing.

## Requirements

- Python 3.8+
- Numba
- NumPy
- CUDA-compatible GPU with appropriate drivers and CUDA toolkit installed

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/bitlocker-brute-force.git
    cd bitlocker-brute-force
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the script:

    ```bash
    python script.py
    ```

2. Follow the prompts to select a drive and GPU.
3. The script will check if the selected drive is encrypted with BitLocker and, if so, will start the brute force process.

## Disclaimer

This script is provided for educational purposes only. Unauthorized attempts to access encrypted data are illegal. Use this script responsibly and ensure you have the right to access the data.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
