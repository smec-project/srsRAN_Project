"""Label all text files using pets_train_data_label.py script.

This module provides functionality to batch process text files for pet data
labeling. It can handle files from multiple directories and automatically
determines whether to use the --base-dir parameter based on the presence of
corresponding eval_data folders.
"""

import os
import subprocess
from pathlib import Path
from typing import Set, List, Optional


def execute_pets_train_data_label(
    file_path: Path, use_base_dir: bool = False
) -> bool:
    """Execute the pets_train_data_label.py script for a given file.

    Args:     file_path: Path to the text file to be processed. use_base_dir:
    Boolean flag to indicate if --base-dir parameter should be used.

    Returns:     True if processing succeeded, False otherwise.
    """
    cmd = ["python3", "pets_train_data_label.py", str(file_path)]
    if use_base_dir:
        cmd.extend(["--base-dir", "eval_data"])

    try:
        subprocess.run(cmd, check=True, cwd=Path.cwd())
        print(f"Successfully processed: {file_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error processing file {file_path}: {e}")
        return False


def find_txt_files_in_directories(search_dirs: List[str]) -> List[Path]:
    """Find all .txt files in the specified directories.

    Args:     search_dirs: List of directory paths to search for txt files.

    Returns:     List of Path objects for all found .txt files.
    """
    txt_files = []

    for search_dir in search_dirs:
        dir_path = Path(search_dir)
        if dir_path.exists() and dir_path.is_dir():
            # Find all .txt files recursively in the directory
            txt_files.extend(dir_path.rglob("*.txt"))
            print(
                f"Found {len(list(dir_path.rglob('*.txt')))} txt files in"
                f" {search_dir}"
            )
        else:
            print(
                f"Warning: Directory {search_dir} does not exist or is not a"
                " directory"
            )

    return txt_files


def get_eval_data_folders() -> Set[str]:
    """Get all folder names in the eval_data directory.

    Returns:     Set of folder names in eval_data directory.
    """
    eval_data_path = Path("eval_data")
    if not eval_data_path.exists():
        print("Warning: eval_data directory does not exist")
        return set()

    return {
        folder.name for folder in eval_data_path.iterdir() if folder.is_dir()
    }


def should_use_base_dir(txt_file: Path, eval_folders: Set[str]) -> bool:
    """Determine if a txt file should use the --base-dir parameter.

    Args:
    txt_file: Path to the text file.
    eval_folders: Set of folder names in eval_data directory.

    Returns:
    True if the file should use --base-dir parameter, False otherwise.
    """
    file_stem = txt_file.stem  # Get filename without extension
    return file_stem in eval_folders


def process_all_txt_files(
    search_directories: Optional[List[str]] = None,
) -> None:
    """Process all txt files found in the specified directories.

    Args:     search_directories: List of directories to search. If None,
    defaults to current directory.
    """
    if search_directories is None:
        search_directories = ["."]

    # Get all eval_data folders
    eval_folders = get_eval_data_folders()

    # Find all txt files in specified directories
    all_txt_files = find_txt_files_in_directories(search_directories)

    if not all_txt_files:
        print("No txt files found in the specified directories")
        return

    print(f"Found {len(all_txt_files)} txt files to process")

    # Process files with base_dir first, then others
    base_dir_files = []
    regular_files = []

    for txt_file in all_txt_files:
        if should_use_base_dir(txt_file, eval_folders):
            base_dir_files.append(txt_file)
        else:
            regular_files.append(txt_file)

    # Process base_dir files first
    if base_dir_files:
        print(
            f"\nProcessing {len(base_dir_files)} base files with --base-dir"
            " parameter:"
        )
        for txt_file in base_dir_files:
            print(f"Processing base file: {txt_file}")
            execute_pets_train_data_label(txt_file, use_base_dir=True)

    # Process regular files
    if regular_files:
        print(f"\nProcessing {len(regular_files)} regular files:")
        for txt_file in regular_files:
            print(f"Processing file: {txt_file}")
            execute_pets_train_data_label(txt_file, use_base_dir=False)


def main():
    """Main function to orchestrate the txt file processing.

    You can modify the search_dirs list to include additional directories where
    txt files might be located.
    """
    # Configure directories to search for txt files
    # Add more directories as needed
    search_dirs = [
        ".",  # Current directory
        "raw_data",  # Raw data directory
        # Add more directories here as needed
    ]

    print("Starting batch processing of txt files...")
    process_all_txt_files(search_dirs)
    print("Batch processing completed.")


if __name__ == "__main__":
    main()
