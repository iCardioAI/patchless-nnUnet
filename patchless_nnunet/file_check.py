from pathlib import Path

import hydra
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(version_base="1.3", config_path="configs", config_name="file_check")
def check_files(cfg: DictConfig):
    data_path = cfg.data_path + cfg.dataset_name

    df = pd.read_csv(data_path + '/' + cfg.csv_file_name, index_col=0)
    df = df[df['valid_segmentation'] == True]

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        r = row.to_dict()

        # check for img
        img_path = Path(data_path) / 'img' / r['study'] / r['view'].lower() / (r['dicom_uuid'] + "_0000.nii.gz")
        if not img_path.is_file():
            raise Exception(f"FILE {img_path} IS MISSING")

        # check for img
        seg_path = Path(data_path) / 'segmentation' / r['study'] / r['view'].lower() / (r['dicom_uuid'] + ".nii.gz")
        if not seg_path.is_file():
            raise Exception(f"FILE {seg_path} IS MISSING")

    print("NO FILES ARE MISSING!")
    print(f"TOTAL NUMBER OF VALID SEQUENCES: {len(df)}")


def main():
    load_dotenv()
    check_files()


if __name__ == "__main__":
    main()
