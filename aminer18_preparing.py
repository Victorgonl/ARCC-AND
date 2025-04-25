# aminer18_preparing.py

import json
import os
import sys

import pandas as pd
from tqdm import tqdm


def main(dataset_dir):
    print("Aminer-18 Dataset Preparing\n")

    print("Loading raw data...")
    aminer18_raw = json.load(open(f"{dataset_dir}/pubs_raw.json"))
    print("Number of documents (papers):", len(aminer18_raw))
    print()

    print("Preparing train data...")
    train = json.load(open(f"{dataset_dir}/name_to_pubs_train_500.json"))
    aminer18_train_map = {
        "split": [],
        "author_name": [],
        "author_id": [],
        "author_uid": [],
        "author_index": [],
        "paper_id": [],
    }
    for name in tqdm(train, "Creating train instances"):
        for author_id in train[name]:
            for paper in train[name][author_id]:
                paper_id, author_index = paper.split("-")
                aminer18_train_map["split"].append("train")
                aminer18_train_map["author_name"].append(name)
                aminer18_train_map["author_uid"].append(f"{name}-{author_id}")
                aminer18_train_map["author_id"].append(author_id)
                aminer18_train_map["paper_id"].append(paper_id)
                aminer18_train_map["author_index"].append(int(author_index))
    aminer18_train_map = pd.DataFrame(
        aminer18_train_map, index=list(range(len(aminer18_train_map["split"])))
    )
    aminer18_train_map.to_csv(
        f"{dataset_dir}/train_instances_500.csv", index_label="index"
    )
    train_instances_500 = pd.read_csv(f"{dataset_dir}/train_instances_500.csv")
    train_instances_500_papers_ids = set(train_instances_500["paper_id"].tolist())
    train_500_raw = {}
    for paper_id in tqdm(aminer18_raw, desc="Creating train raw data"):
        if paper_id in train_instances_500_papers_ids:
            train_500_raw[paper_id] = aminer18_raw[paper_id]
    print("Saving train...")
    json.dump(
        train_500_raw, open(f"{dataset_dir}/pubs_raw_train_500.json", "w"), indent=4
    )
    print()

    print("Preparing test data...")
    test = json.load(open(f"{dataset_dir}/name_to_pubs_test_100.json"))
    aminer18_test_map = {
        "split": [],
        "author_name": [],
        "author_id": [],
        "author_uid": [],
        "author_index": [],
        "paper_id": [],
    }
    for name in tqdm(test, "Creating test instances"):
        for author_id in test[name]:
            for paper in test[name][author_id]:
                paper_id, author_index = paper.split("-")
                aminer18_test_map["split"].append("test")
                aminer18_test_map["author_name"].append(name)
                aminer18_test_map["author_uid"].append(f"{name}-{author_id}")
                aminer18_test_map["paper_id"].append(paper_id)
                aminer18_test_map["author_id"].append(author_id)
                aminer18_test_map["author_index"].append(int(author_index))
    aminer18_test_map = pd.DataFrame(
        aminer18_test_map,
        index=list(
            range(
                len(aminer18_train_map["split"]),
                len(aminer18_test_map["split"]) + len(aminer18_train_map["split"]),
            )
        ),
    )
    aminer18_test_map.to_csv(
        f"{dataset_dir}/test_instances_100.csv", index_label="index"
    )
    test_instances_100 = pd.read_csv(f"{dataset_dir}/test_instances_100.csv")
    test_instances_100_papers_ids = set(test_instances_100["paper_id"].tolist())
    test_100_raw = {}
    for paper_id in tqdm(aminer18_raw, desc="Creating test raw data"):
        if paper_id in test_instances_100_papers_ids:
            test_100_raw[paper_id] = aminer18_raw[paper_id]
    print("Saving test...")
    json.dump(
        test_100_raw, open(f"{dataset_dir}/pubs_raw_test_100.json", "w"), indent=4
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python aminer18_preparing.py <dataset_folder>")
        sys.exit(1)
    dataset_dir = sys.argv[1]
    main(dataset_dir)
