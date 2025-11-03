import argparse
import os
import shutil
from pathlib import Path
from typing import Literal

import pandas as pd
from tqdm import tqdm

MALE_LABEL, FEMALE_LABEL = 'male_masculine', 'female_feminine'
PATH_ERROR = \
'''
Haven't input dataset absolute path

    Try run:
        uv run python data_labeling.py -p <your-dataset-absolute-path>

path like: xxx/cv-corpus-22.0-2025-06-20/en
'''

PATH_VALID_ERROR = \
'''
Error by files not found(clips, train.tsv): Please input valid "Common Voices dataset" absolute path,
you can download dataset at https://commonvoice.mozilla.org/

path like: xxx/cv-corpus-22.0-2025-06-20/en
'''

label_simplifier = {MALE_LABEL: 'male', FEMALE_LABEL: 'female'}
"""Simplifier source dataset label to project label."""


def _voices_labeling_process(
    output_dir: Path,
    voices_tsv: Path,
    clips_dir: Path,
    count: int = 1000,
    tag: Literal['Train', 'Test'] = 'train',
) -> str:
    """ Process data labeling. (處理資料標註)
    :param output_path: like '/data/xxx'
    :param count: labeling count.
    :return: result logs.
    """
    male_voices_dir = output_dir / 'male_voices'
    female_voices_dir = output_dir / 'female_voices'
    for d in (male_voices_dir, female_voices_dir):
        d.mkdir(parents=True, exist_ok=True)

    voices_detail = pd.read_csv(voices_tsv, sep='\t', low_memory=False)
    series_filter = voices_detail['gender'].isin([MALE_LABEL, FEMALE_LABEL])
    filtered_train_data = voices_detail[series_filter]

    # Process voices
    male_voices_count = female_voices_count = 0
    result: list[tuple[str, str]] = []
    for idx, row in tqdm(filtered_train_data.iterrows(), total=count, desc=f'{tag} data labeling'):
        # Check source file exist
        source_file_path = clips_dir / row['path']
        if not source_file_path.exists():
            print(f"File dose not exist, jump：{source_file_path}")
            continue

        # Voice gender check
        if row['gender'] not in [MALE_LABEL, FEMALE_LABEL]: continue
        if male_voices_count + female_voices_count >= count: break
        if row['gender'] == MALE_LABEL:
            if male_voices_count >= count // 2: continue
            male_voices_count += 1
            current_output_dir = male_voices_dir
        else:
            if female_voices_count >= count // 2: continue
            female_voices_count += 1
            current_output_dir = female_voices_dir

        # Make new file path
        base, ext = os.path.splitext(row['path'])
        gender = label_simplifier.get(row['gender'])
        new_filename = f'{base}_{gender}{ext}'
        destination_file_path = current_output_dir / new_filename

        try:
            shutil.copy(source_file_path, destination_file_path)
            result.append((row['path'], row['gender']))
        except Exception as e:
            print(f"Error when copy '{source_file_path}'：{e}")

    return f'Data labeling completed, check {output_dir}/'


def _main(input_source_path: str | None = None, count: int = 1000) -> str:
    """ Main flow. (主要工作流)
    :param input_source_path: Absolute path of dataset.
    :param count: Max process count of data, male and female 50%/50%.
    :return: log message.
    """

    # Read source file
    if input_source_path is None: return PATH_ERROR
    voices_dir = Path(input_source_path)
    clips_dir = voices_dir / 'clips'
    train_tsv = voices_dir / 'train.tsv'
    test_tsv = voices_dir / 'test.tsv'

    # Check path valid
    check_list = (voices_dir, clips_dir, train_tsv, test_tsv)
    if not all(p.exists() for p in check_list): return PATH_VALID_ERROR

    _voices_labeling_process(output_dir=Path('data/train'),
                             voices_tsv=train_tsv, clips_dir=clips_dir, count=count, tag='Train')
    _voices_labeling_process(output_dir=Path('data/test'),
                             voices_tsv=test_tsv, clips_dir=clips_dir, count=count, tag='Test')

    # output_train_tsv(result)
    return f'All completed!'


def output_train_tsv(data: list[tuple[str, str]]) -> None:
    pd.DataFrame({
        'path': [path for path, _ in data],
        'gender': [gender for _, gender in data],
    }).to_csv('train.tsv', sep='\t', header=True, index=False, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Labeling voice data and classify to folder.')
    parser.add_argument('-p', '--path', type=str, required=True, help='Absolute path of dataset.')
    parser.add_argument('-c', '--count', type=int, default=1000,
                        help='Max process count of data. (default is 1000)')
    args = parser.parse_args()

    message = _main(input_source_path=args.path, count=args.count)
    print(message)