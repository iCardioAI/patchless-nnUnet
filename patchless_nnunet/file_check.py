import multiprocessing
from pathlib import Path

import hydra
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig
from tqdm import tqdm


def check_file(in_tuple):
    data_path, r = in_tuple
    # check for img
    img_path = Path(data_path) / 'img' / r['study'] / r['view'].lower() / (r['dicom_uuid'] + "_0000.nii.gz")
    if not img_path.is_file():
        return False, img_path

    # check for img
    seg_path = Path(data_path) / 'segmentation' / r['study'] / r['view'].lower() / (r['dicom_uuid'] + ".nii.gz")
    if not seg_path.is_file():
        return False, seg_path
    return True, r['dicom_uuid']


@hydra.main(version_base="1.3", config_path="configs", config_name="file_check")
def check_files(cfg: DictConfig):
    data_path = cfg.data_path + '/' + cfg.dataset_name

    df = pd.read_csv(data_path + '/' + cfg.csv_file_name, index_col=0)
    df = df[df['valid_segmentation'] == True]
    items = [(data_path, r[1].to_dict()) for r in df.iterrows()]
    missing = 0
    with multiprocessing.Pool() as pool:
        for result in tqdm(pool.map(check_file, items), total=len(df)):
            if not result[0]:
                missing += 1
                print(f"{result[1]} IS MISSING!")

    print(f"{missing} FILES ARE MISSING!")
    print(f"TOTAL NUMBER OF VALID SEQUENCES: {len(df) - missing}")


def main():
    load_dotenv()
    check_files()


if __name__ == "__main__":
    main()
