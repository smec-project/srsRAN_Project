import os
import subprocess
from pathlib import Path


def run_pets_train_data_label(file_path, base_dir=False):
    """
    Execute pets_train_data_label.py script
    Args:
        file_path: Path to the txt file
        base_dir: Boolean flag to indicate if --base-dir parameter should be used
    """
    cmd = ["python3", "pets_train_data_label.py", str(file_path)]
    if base_dir:
        cmd.extend(["--base-dir", "eval_data"])

    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully processed: {file_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing file {file_path}: {e}")


def main():
    # Check if eval_data directory exists
    eval_data_path = Path("eval_data")
    if not eval_data_path.exists():
        print("Error: eval_data directory does not exist")
        return

    # First process all base files
    eval_folders = [f for f in eval_data_path.iterdir() if f.is_dir()]
    processed_base_files = set()

    for folder in eval_folders:
        folder_name = folder.name
        base_txt = Path(f"./{folder_name}.txt")

        if base_txt.exists():
            print(f"Processing base file: {base_txt}")
            run_pets_train_data_label(base_txt, base_dir=True)
            processed_base_files.add(base_txt.name)

    # Process all txt files in current directory except those already processed
    print("\nProcessing remaining txt files...")
    for txt_file in Path(".").glob("*.txt"):
        if txt_file.name not in processed_base_files:
            print(f"Processing file: {txt_file}")
            run_pets_train_data_label(txt_file)


if __name__ == "__main__":
    main()
