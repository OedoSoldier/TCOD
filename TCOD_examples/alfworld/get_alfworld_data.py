"""
We use this script to create the huggingface format dataset files for the alfworld dataset.
NOTE: You need to install the alfworld dataset in first: https://github.com/alfworld/alfworld
"""
import argparse
import glob
import json
import os
import random

random.seed(42)


OLD_ALFWORLD_PREFIX = "/nas/wjq/alfworld"


def _records_from_files(game_files):
    return [{"game_file": game_file_path, "target": ""} for game_file_path in game_files]


def _write_jsonl(output_file, data):
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def _sample_files(game_files, size, split_name):
    if size is None:
        size = len(game_files)
    assert size <= len(game_files), f"{split_name}_size {size} > available {len(game_files)}"
    return random.sample(game_files, size)


def _localize_existing_split(filepath, game_data_path):
    if not os.path.exists(filepath):
        return None

    data = []
    game_data_path = os.path.abspath(os.path.expanduser(game_data_path)).rstrip("/")
    with open(filepath, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            game_file = item.get("game_file", "")
            if game_file.startswith(OLD_ALFWORLD_PREFIX):
                game_file = game_data_path + game_file[len(OLD_ALFWORLD_PREFIX):]
                item["game_file"] = game_file
            data.append(item)
    return data


def create_dataset_files(
    game_data_path,
    output_dir,
    train_size=None,
    test_size=None,
    test_unseen_size=None,
):
    # get all matched game files from train and eval directories
    train_game_files = glob.glob(
        os.path.expanduser(f"{game_data_path}/json_2.1.1/train/*/*/game.tw-pddl")
    )
    test_seen_game_files = glob.glob(
        os.path.expanduser(f"{game_data_path}/json_2.1.1/valid_seen/*/*/game.tw-pddl")
    )
    test_unseen_game_files = glob.glob(
        os.path.expanduser(f"{game_data_path}/json_2.1.1/valid_unseen/*/*/game.tw-pddl")
    )

    # get absolute path
    train_game_files = [os.path.abspath(file) for file in train_game_files]
    test_seen_game_files = [os.path.abspath(file) for file in test_seen_game_files]
    test_unseen_game_files = [os.path.abspath(file) for file in test_unseen_game_files]
    train_game_files = sorted(train_game_files)
    test_seen_game_files = sorted(test_seen_game_files)
    test_unseen_game_files = sorted(test_unseen_game_files)

    print(f"Total train game files found: {len(train_game_files)}")
    print(f"Total seen eval game files found: {len(test_seen_game_files)}")
    print(f"Total unseen eval game files found: {len(test_unseen_game_files)}")

    # randomly select the game files
    selected_train_files = _sample_files(train_game_files, train_size, "train")
    selected_test_seen_files = _sample_files(test_seen_game_files, test_size, "test")
    selected_test_unseen_files = _sample_files(
        test_unseen_game_files,
        test_unseen_size,
        "test_unseen",
    )

    # make the output directory
    os.makedirs(output_dir, exist_ok=True)

    # create train and test data
    train_data = _records_from_files(selected_train_files)
    test_seen_data = _records_from_files(selected_test_seen_files)
    test_unseen_data = _records_from_files(selected_test_unseen_files)
    hard_path = os.path.join(output_dir, "train_hard.jsonl")
    hard_data = _localize_existing_split(hard_path, game_data_path)

    # create dataset_dict
    dataset_dict = {
        "train": train_data,
        # Keep the original TCOD filename for seen eval compatibility.
        "test": test_seen_data,
        "test_unseen": test_unseen_data,
    }
    if hard_data is not None:
        dataset_dict["train_hard"] = hard_data

    for split, data in dataset_dict.items():
        output_file = os.path.join(output_dir, f"{split}.jsonl")
        _write_jsonl(output_file, data)

    # create dataset_dict.json
    dataset_info = {
        "citation": "",
        "description": "Custom dataset",
        "splits": {
            split: {"name": split, "num_examples": len(data)}
            for split, data in dataset_dict.items()
        },
    }

    with open(os.path.join(output_dir, "dataset_dict.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)

    hard_msg = f", {len(hard_data)} hard" if hard_data is not None else ""
    print(
        "Created dataset with "
        f"{len(train_data)} train, {len(test_seen_data)} seen eval, "
        f"{len(test_unseen_data)} unseen eval{hard_msg} examples."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_data_path", type=str, default=None, required=False)
    parser.add_argument("--local_dir", type=str, default=None, required=False)
    parser.add_argument("--train_size", type=int, default=None, required=False)
    parser.add_argument("--test_size", type=int, default=None, required=False)
    parser.add_argument("--test_unseen_size", type=int, default=None, required=False)
    args = parser.parse_args()

    if args.game_data_path is None:
        # ALFWORLD_DATA is the dataset path in the environment variable
        # you need to set it when install alfworld dataset
        from alfworld.info import ALFWORLD_DATA

        args.game_data_path = ALFWORLD_DATA

    if args.local_dir is None:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        args.local_dir = f"{current_file_dir}/alfworld_data"

    # use all data by default, or specify train_size and test_size if needed
    create_dataset_files(
        game_data_path=args.game_data_path,
        output_dir=args.local_dir,
        train_size=args.train_size,
        test_size=args.test_size,
        test_unseen_size=args.test_unseen_size,
    )
