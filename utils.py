import random
from collections.abc import Callable

import torch
from torch.utils.data import Dataset


class ContrastiveDataset(Dataset):
    data: list[tuple[any, any]]
    classes: list[any]
    class_indices: dict[any, list[any]]
    similarity_target_transform: Callable

    def __init__(
        self,
        base_dataset_class: type[Dataset],
        *args,
        similarity_target_transform: Callable = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.data, self.classes, self.class_indices = collect_and_group_data(
            base_dataset_class(*args, **kwargs)
        )
        self.similarity_target_transform = similarity_target_transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index1: int) -> tuple[any, any, any]:
        x1, label1 = self.data[index1]
        x2: any
        similarity: any

        # even index => 2nd image from the same class
        if index1 % 2 == 0:
            index2 = index1
            while index2 == index1:
                index2 = random.choice(self.class_indices[label1])
            x2 = self.data[index2][0]
            similarity = 1

        # odd index => 2nd image from a different class
        else:
            label2 = label1
            while label2 == label1:
                x2, label2 = random.choice(self.data)
            similarity = 0

        if self.similarity_target_transform:
            similarity = self.similarity_target_transform(similarity)
        elif isinstance(x1, torch.Tensor):
            similarity = torch.tensor(similarity, dtype=x1.dtype)

        return x1, x2, similarity


def collect_and_group_data(
    dataset: Dataset,
) -> tuple[list[tuple[any, any]], list[any], dict[any, list[any]]]:
    data: list[tuple[any, any]] = []
    classes: set[any] = set()
    class_indices: dict[any, list[any]] = {}
    index = 0
    for x, y in dataset:
        data.append((x, y))
        classes.add(y)
        class_indices.setdefault(y, []).append(index)
        index += 1
    return data, list(classes), class_indices
