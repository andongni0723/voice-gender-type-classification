import os
import random
from pathlib import Path
from typing import Literal

from tqdm import tqdm


def random_pick(path_obj: Path, count: int) -> list[str]:
    """
    :param path_obj: folder path which pick files.
    :param count: files count will pick.
    :return: Back files name list or error message.
    """
    try:
        total_len = len(list(path_obj.iterdir()))
        title = 'Reading voices file in folder.'
        all_files = [p for p in tqdm(path_obj.iterdir(), desc=title, total=total_len) if p.is_file()]
        if len(all_files) < count:
            print(f'Files count in {path_obj} less then {count}. Will return all files in folder.')
            return all_files
        return random.sample(all_files, count)

    except FileNotFoundError:
        print(f'Folder Not Found : {path_obj}')
        return []

    except Exception as e:
        print(f'Unknown Error: {e}')
        return []


def import_voice_with_filename_and_label(
    data_path: str,
    count_total: int = 100,
    verbose: bool = False
) -> list[tuple[str, Literal['male', 'female']]]:
    """
    :param data_path: the voice data folder (like data/train)
    :param count_total: max import count.
    :param verbose: output debug message
    :return: Back a list with tuples have (file path, label)
    """
    res: list[tuple[str, Literal['male', 'female']]] = []
    male_data = random_pick(Path(data_path) / 'male_voices', count_total // 2)
    female_data = random_pick(Path(data_path) / 'female_voices', count_total // 2)
    for filename in tqdm(male_data + female_data, total=count_total, desc='Import Voices'):
        name, _ = os.path.splitext(filename) # a_male.mp3 -> ('a_male', '.mp3')
        label = name.split('_')[-1] # 'male'
        if label not in ('male', 'female'): continue
        res.append((filename, label))

    if verbose:
        print(res[:10])
        print('...')
    return res

def gener_trans(label: str) -> str:
    match label:
        case 'man': return 'male'
        case 'girl': return 'female'
        case _: return label

if __name__ == '__main__':
    import_voice_with_filename_and_label('../data/train', verbose=True)