# aminer18_preparing.py

import json
import os
import sys

import pandas as pd
from tqdm import tqdm


def main(dataset_dir):
    print("# Aminer-18 Dataset Preparing\n")

    # Load raw publications
    with open(os.path.join(dataset_dir, "pubs_raw.json")) as f:
        aminer18_raw = json.load(f)
    print("Number of documents (papers):", len(aminer18_raw))

    # Load train data
    train = json.load(open(os.path.join(dataset_dir, "name_to_pubs_train_500.json")))

    aminer18_train_map = {
        "split": [],
        "author_name": [],
        "author_id": [],
        "author_uid": [],
        "author_index": [],
        "paper_id": [],
    }

    for name in tqdm(train, desc="Processing train authors"):
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

    train_csv_path = os.path.join(dataset_dir, "train_instances_500.csv")
    aminer18_train_map.to_csv(train_csv_path, index_label="index")

    pubs_raw_train_500 = {
        paper_id: aminer18_raw[paper_id]
        for paper_id in tqdm(aminer18_raw, desc="Filtering raw train papers")
        if paper_id in aminer18_train_map["paper_id"].values
    }

    json.dump(
        pubs_raw_train_500,
        open(os.path.join(dataset_dir, "pubs_raw_train_500.json"), "w"),
        indent=4,
    )

    # Load test data
    test = json.load(open(os.path.join(dataset_dir, "name_to_pubs_test_100.json")))

    aminer18_test_map = {
        "split": [],
        "author_name": [],
        "author_id": [],
        "author_uid": [],
        "author_index": [],
        "paper_id": [],
    }

    for name in tqdm(test, desc="Processing test authors"):
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
                len(aminer18_train_map),
                len(aminer18_test_map["split"]) + len(aminer18_train_map),
            )
        ),
    )

    test_csv_path = os.path.join(dataset_dir, "test_instances_100.csv")
    aminer18_test_map.to_csv(test_csv_path, index_label="index")

    pubs_raw_test_100 = {
        paper_id: aminer18_raw[paper_id]
        for paper_id in tqdm(aminer18_raw, desc="Filtering raw test papers")
        if paper_id in aminer18_test_map["paper_id"].values
    }

    json.dump(
        pubs_raw_test_100,
        open(os.path.join(dataset_dir, "pubs_raw_test_100.json"), "w"),
        indent=4,
    )

    # Report stats
    train_names = aminer18_train_map["author_name"].unique().tolist()
    test_names = aminer18_test_map["author_name"].unique().tolist()
    common_names = set(train_names).intersection(set(test_names))
    print("Number of names both in train and test:", len(common_names))

    train_authors = aminer18_train_map["author_uid"].unique().tolist()
    test_authors = aminer18_test_map["author_uid"].unique().tolist()
    common_authors = set(train_authors).intersection(set(test_authors))
    print("Number of authors both in train and test:", len(common_authors))

    train_papers = aminer18_train_map["paper_id"].unique().tolist()
    test_papers = aminer18_test_map["paper_id"].unique().tolist()
    common_papers = set(train_papers).intersection(set(test_papers))
    print("Number of papers both in train and test:", len(common_papers))

    aminer18_all = pd.concat([aminer18_train_map, aminer18_test_map])

    print("Number of entities:", len(aminer18_all))
    print("Number of names:", len(aminer18_all["author_name"].unique()))
    print("Number of authors:", len(aminer18_all["author_uid"].unique()))
    print(
        "Number of first authors:",
        len(aminer18_all[aminer18_all["author_index"] == 0]["author_uid"].unique()),
    )
    print("Number of papers:", len(aminer18_all["paper_id"].unique()))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python aminer18_preparing.py <dataset_folder>")
        sys.exit(1)
    dataset_dir = sys.argv[1]
    main(dataset_dir)
