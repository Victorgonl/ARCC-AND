# aminer18_preparing.py

import argparse
import json
from pathlib import Path

import pandas as pd


def main(dataset_dir):
    dataset_dir = Path(dataset_dir)

    # Load raw publications
    aminer18_raw = json.load(open(dataset_dir / "pubs_raw.json"))
    print("Number of documents (papers):", len(aminer18_raw))

    # Load training data
    train = json.load(open(dataset_dir / "name_to_pubs_train_500.json"))
    aminer18_train_map = {
        "split": [],
        "author_name": [],
        "author_id": [],
        "author_uid": [],
        "author_index": [],
        "paper_id": [],
    }

    for name in train:
        for author_id in train[name]:
            for paper in train[name][author_id]:
                paper_id, author_index = paper.split("-")
                aminer18_train_map["split"].append("train")
                aminer18_train_map["author_name"].append(name)
                aminer18_train_map["author_uid"].append(f"{name}-{author_id}")
                aminer18_train_map["author_id"].append(author_id)
                aminer18_train_map["paper_id"].append(paper_id)
                aminer18_train_map["author_index"].append(int(author_index))

    aminer18_train_map = pd.DataFrame(aminer18_train_map)
    aminer18_train_map.to_csv(
        dataset_dir / "train_instances_500.csv", index_label="index"
    )

    pubs_raw_train_500 = {
        pid: aminer18_raw[pid]
        for pid in aminer18_raw
        if pid in aminer18_train_map["paper_id"].values
    }
    json.dump(
        pubs_raw_train_500, open(dataset_dir / "pubs_raw_train_500.json", "w"), indent=4
    )

    # Load test data
    test = json.load(open(dataset_dir / "name_to_pubs_test_100_original.json"))
    aminer18_test_map = {
        "split": [],
        "author_name": [],
        "author_id": [],
        "author_uid": [],
        "author_index": [],
        "paper_id": [],
    }

    for name in test:
        for author_id in test[name]:
            for paper in test[name][author_id]:
                paper_id, author_index = paper.split("-")
                aminer18_test_map["split"].append("test")
                aminer18_test_map["author_name"].append(name)
                aminer18_test_map["author_uid"].append(f"{name}-{author_id}")
                aminer18_test_map["author_id"].append(author_id)
                aminer18_test_map["paper_id"].append(paper_id)
                aminer18_test_map["author_index"].append(int(author_index))

    aminer18_test_map = pd.DataFrame(aminer18_test_map)
    aminer18_test_map.index += len(aminer18_train_map)
    aminer18_test_map.to_csv(
        dataset_dir / "test_instances_100.csv", index_label="index"
    )

    pubs_raw_train_100 = {
        pid: aminer18_raw[pid]
        for pid in aminer18_raw
        if pid in aminer18_test_map["paper_id"].values
    }
    json.dump(
        pubs_raw_train_100, open(dataset_dir / "pubs_raw_train_100.json", "w"), indent=4
    )

    # Summary
    train_names = aminer18_train_map["author_name"].unique().tolist()
    test_names = aminer18_test_map["author_name"].unique().tolist()
    print(
        "Number of names both in train and test:",
        len(set(train_names) & set(test_names)),
    )

    train_authors = aminer18_train_map["author_uid"].unique().tolist()
    test_authors = aminer18_test_map["author_uid"].unique().tolist()
    print(
        "Number of authors both in train and test:",
        len(set(train_authors) & set(test_authors)),
    )

    train_papers = aminer18_train_map["paper_id"].unique().tolist()
    test_papers = aminer18_test_map["paper_id"].unique().tolist()
    print(
        "Number of papers both in train and test:",
        len(set(train_papers) & set(test_papers)),
    )

    # Combine all
    aminer18_all = pd.concat([aminer18_train_map, aminer18_test_map])
    print("Number of entities:", len(aminer18_all))
    print("Number of names:", len(aminer18_all["author_name"].unique()))
    print("Number of authors:", len(aminer18_all["author_uid"].unique()))
    print(
        "Number of first authors:",
        len(aminer18_all[aminer18_all["author_index"] == 0]["author_uid"].unique()),
    )
    print("Number of papers:", len(aminer18_all["paper_id"].unique()))

    # Filter and dump for other sets
    for subset in [
        ("train_instances_400.csv", "pubs_raw_train_400.json"),
        ("validation_instances_100.csv", "pubs_raw_val_100.json"),
        ("test_instances.csv", "pubs_raw_test_100.json"),
    ]:
        csv_file, json_out = subset
        instances = pd.read_csv(dataset_dir / csv_file)
        paper_ids = set(instances["paper_id"].tolist())
        raw_subset = {
            pid: aminer18_raw[pid] for pid in aminer18_raw if pid in paper_ids
        }
        json.dump(raw_subset, open(dataset_dir / json_out, "w"), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Aminer-18 dataset")
    parser.add_argument("dataset_dir", help="Path to the Aminer-18 dataset directory")
    args = parser.parse_args()

    main(args.dataset_dir)
