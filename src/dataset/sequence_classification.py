from abc import ABC, abstractmethod

import pandas as pd
from datasets import Dataset, load_dataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from datasets import DatasetDict


class BaseDataset(ABC):
    """Abstract class for dataset."""

    @abstractmethod
    def name(self) -> str:
        """Return name of dataset."""

    @abstractmethod
    def num_classes(self) -> int:
        """Return number of classes in dataset."""

    @abstractmethod
    def load(self) -> DatasetDict:
        """Function to load and preprocess dataset."""


class ToxigenDataset(BaseDataset):
    def name(self) -> str:
        """Return name of dataset."""
        return "toxigen"

    @staticmethod
    def label_annotations(annotated: pd.DataFrame) -> pd.DataFrame:
        """Create toxic labeles based on score of model and human labels."""
        # Annotations should be the annotated dataset
        label = ((annotated.toxicity_ai + annotated.toxicity_human) > 5.5).astype(int)
        labeled_annotations = pd.DataFrame()
        labeled_annotations["text"] = annotated.text.tolist()
        labeled_annotations["label"] = label
        return labeled_annotations

    def num_classes(self) -> int:
        """Return number of classes in dataset."""
        return 2

    def load(self) -> DatasetDict:
        """Load dataset, preprocess and return train test split."""
        data = load_dataset("skg/toxigen-data", name="annotated")

        train = pd.DataFrame(data["train"])
        test = pd.DataFrame(data["test"])

        train = self.label_annotations(train)
        train = Dataset.from_dict(
            {"text": train.text.values, "label": train.label.values}
        )
        test = self.label_annotations(test)
        test = Dataset.from_dict(
            {"text": test.text.values, "label": test.label.values}
        )

        split_dataset = train.train_test_split(0.2, seed=42)
        train, validation = split_dataset['train'], split_dataset['test']

        datasets = DatasetDict(
            {
                "train": train,
                "validation": validation,
                "test": test,
            }
        )
        return datasets


class JigsawDataset(BaseDataset):
    def __init__(self, data_path):
        self.data_path = data_path

    def name(self) -> str:
        """Return name of dataset."""
        return "jigsaw"

    def num_classes(self) -> int:
        """Return number of classes in dataset."""
        return 2

    def load(self):
        """Load dataset, preprocess and return train test split."""
        train = pd.read_csv(self.data_path)
        train["label"] = (train[train.columns[2:]].sum(1) > 0) * 1
        train_sort = train.sort_values("label")

        x_train, x_test, y_train, y_test = train_test_split(
            train_sort.comment_text.values,
            train_sort.label.values,
            stratify=train_sort.label.values,
            test_size=0.1,
            random_state=42,
        )

        datasets = DatasetDict(
            {
                "train": Dataset.from_dict({"text": x_train, "label": y_train}),
                "validation": Dataset.from_dict({"text": x_test, "label": y_test}),
            }
        )
        return datasets


class ImplicitHateDataset(BaseDataset):
    """https://huggingface.co/datasets/SALT-NLP/ImplicitHate"""

    def __init__(self, data_path):
        self.data_path = data_path

    def name(self) -> str:
        """Return name of dataset."""
        return "implicit_hate"

    def num_classes(self) -> int:
        """Return number of classes in dataset."""
        return 3

    def load(self):
        """Load dataset, preprocess and return train test split."""
        data = pd.read_csv(self.data_path, sep="\t")

        data["label"] = 0
        data.loc[data["class"] == "implicit_hate", "label"] = 1
        data.loc[data["class"] == "explicit_hate", "label"] = 2

        x_train, x_test, y_train, y_test = train_test_split(
            data["post"].values,
            data["label"].values,
            stratify=data["label"].values,
            test_size=0.1,
            random_state=42,
        )

        datasets = DatasetDict(
            {
                "train": Dataset.from_dict({"text": x_train, "label": y_train}),
                "validation": Dataset.from_dict({"text": x_test, "label": y_test}),
            }
        )
        return datasets


class CoLa(BaseDataset):
    def name(self) -> str:
        """Return name of dataset."""
        return "cola"

    def num_classes(self) -> int:
        """Return number of classes in dataset."""
        return 2

    def load(self):
        dataset = load_dataset("nyu-mll/glue", "cola")
        del dataset["test"]
        datasets = dataset.remove_columns("idx").rename_column("sentence", "text")

        split_dataset = datasets['train'].train_test_split(0.2, seed=42)
        train, validation = split_dataset['train'], split_dataset['test']
        return DatasetDict({
            'train': train,
            'validation': validation,
            'test': datasets['validation'],
        })


class SST2(BaseDataset):
    def name(self) -> str:
        """Return name of dataset."""
        return "SST2"

    def num_classes(self) -> int:
        """Return number of classes in dataset."""
        return 2

    def load(self) -> DatasetDict:
        """Load dataset, preprocess and return train test split."""
        data = load_dataset("glue", "sst2")
        data = data.rename_column('sentence', 'text').remove_columns('idx')
        split_dataset = data['train'].train_test_split(0.2, seed=42)
        train, validation = split_dataset['train'], split_dataset['test']
        test = data['validation']
        return DatasetDict({
            'train': train,
            "validation": validation,
            "test": test,
        })


class NewsGroups(BaseDataset):
    def name(self) -> str:
        """Return name of dataset."""
        return "20newsgroups"

    def num_classes(self) -> int:
        """Return number of classes in dataset."""
        return 20

    def load(self) -> DatasetDict:
        data = fetch_20newsgroups(subset="train")
        data = {
            "label": data["target"],
            "text": data["data"],
        }
        data = Dataset.from_dict(data)
        split_dataset = data.train_test_split(0.2, seed=42)
        train, validation = split_dataset['train'], split_dataset['test']
        test = fetch_20newsgroups(subset="test")
        test = Dataset.from_dict({"label": test["target"], "text": test["data"]})
        datasets = DatasetDict(
            {
                "train": train,
                "validation": validation,
                "test": test,
            }
        )
        return datasets


class SST5(BaseDataset):
    def name(self) -> str:
        """Return name of dataset."""
        return "SST5"

    def num_classes(self) -> int:
        """Return number of classes in dataset."""
        return 5

    def load(self) -> DatasetDict:
        """Load dataset, preprocess and return train test split."""
        data = load_dataset("SetFit/sst5")
        data = data.remove_columns('label_text')
        return DatasetDict({
            'train': data['train'],
            "validation": data['validation'],
            "test": data['test']
        })
