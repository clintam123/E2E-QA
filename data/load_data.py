from datasets import load_dataset


def load_data():
    # Load the SQuAD dataset
    DATASET_NAME = "squad_v2"
    raw_datasets = load_dataset(DATASET_NAME, split="train")

    # Remove examples with no answers
    raw_datasets = raw_datasets.filter(lambda x: len(x["answers"]["text"]) > 0)
    columns = raw_datasets.column_names
    columns_to_keep = ["id", "context", "question", "answers"]
    columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
    raw_datasets = raw_datasets.remove_columns(columns_to_remove)

    return raw_datasets
