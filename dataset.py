import random
from collections.abc import Iterator
from os import path

import solt
import torch
import torch.utils.data
from PIL import Image
from solt import transforms as slt
from torch import Tensor
from torchvision.datasets import utils
from torchvision.transforms import functional

_SUB_ROOT_DIR = "omniglot-py"
_DOWNLOAD_URL_PREFIX: str = (
    "https://raw.githubusercontent.com/brendenlake/omniglot/master/python"
)
_IMAGES_EVALUATION: str = "images_evaluation.zip"
_IMAGES_BACKGROUND: str = "images_background.zip"
_ZIPS_MD5: dict[str, str] = {
    _IMAGES_BACKGROUND: "68d2efa1b9178cc56df9314c21c6e718",
    _IMAGES_EVALUATION: "6b91aef0f799c5bb55b94e3f2daec811",
}
_NUM_CHARS_PER_ALPH: int = 15  # number of characters to take per alphabet
_SAME_PAIR_TARGET = torch.tensor(1.0)
_DIFF_PAIR_TARGET = torch.tensor(0.0)


class OmniglotForVerificationTask(torch.utils.data.IterableDataset):
    data: list[list[list[Image.Image]]]
    root: str
    train: bool
    n_sample_pairs: int
    augment: bool
    zip_name_no_ext: str
    drawers_slice: slice

    def __init__(
        self,
        root: str,
        train: bool = True,
        num_samples: int = 30_000,
        augment: bool = False,
        download: bool = False,
    ) -> None:
        if isinstance(root, str):
            root = path.expanduser(root)
        self.root = path.join(root, _SUB_ROOT_DIR)
        self.train = train
        assert (
            num_samples > 0 and num_samples % 2 == 0
        ), "n_samples must be a positive even number"
        self.n_sample_pairs = num_samples
        if augment:
            assert train, "augment can only be True for training set"
        self.augment = augment
        self.zip_name_no_ext = _IMAGES_BACKGROUND if train else _IMAGES_EVALUATION
        self.drawers_slice = slice(12) if train else slice(16)
        if download:
            self.download_and_prepare()
        else:
            self.prepare_data()

    def download_and_prepare(self):
        self._download()
        if not self._check_integrity():
            raise RuntimeError(
                """Dataset not found or corrupted. You can use download=True to
                download it"""
            )
        self.prepare_data()

    def prepare_data(self):
        self.data = []
        root = path.join(self.root, self.zip_name_no_ext)
        for alphabet_dir in utils.list_dir(root, prefix=True):
            characters_dirs = utils.list_dir(alphabet_dir, prefix=True)
            if len(characters_dirs) < _NUM_CHARS_PER_ALPH:
                continue

            characters_dirs = random.sample(characters_dirs, k=_NUM_CHARS_PER_ALPH)

            # Take the `drawers_slice`-indexed drawers’ drawings from each of the
            # chosen characters in the alphabet.
            alphabet_data: list[list[Image.Image]] = [
                [
                    Image.open(drawing_path, mode="r").convert("L")
                    for drawing_path in utils.list_files(
                        root=character_dir,
                        suffix=".png",
                        prefix=True,
                    )[self.drawers_slice]
                ]
                for character_dir in characters_dirs
            ]

            self.data.append(alphabet_data)

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
        drawers_range = range(20)[self.drawers_slice]
        times_to_augment = 8 if self.augment else 0

        for _ in range(self.n_sample_pairs // 2):
            characters = random.choice(self.data)  # choose a random alphabet
            char1, char2 = random.sample(characters, 2)  # 2 diff random chars in it
            drawer1, drawer2 = random.sample(drawers_range, 2)  # 2 diff drawers' idxs

            anchor = char1[drawer1]
            positive = char1[drawer2]
            anchor_tensor = to_tensor(anchor)
            yield anchor_tensor, to_tensor(positive), _SAME_PAIR_TARGET

            negative = char2[drawer2]
            yield anchor_tensor, to_tensor(negative), _DIFF_PAIR_TARGET

            for _ in range(times_to_augment):
                anchor_tensor = to_augmented_tensor(anchor)
                yield anchor_tensor, to_augmented_tensor(positive), _SAME_PAIR_TARGET
                yield anchor_tensor, to_augmented_tensor(negative), _DIFF_PAIR_TARGET

    def __len__(self) -> int:
        return self.n_sample_pairs

    def _check_integrity(self) -> bool:
        return utils.check_integrity(
            fpath=path.join(self.root, self.zip_name_no_ext + ".zip"),
            md5=_ZIPS_MD5[self.zip_name_no_ext],
        )

    def _download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        zip_filename = self.zip_name_no_ext + ".zip"
        url = _DOWNLOAD_URL_PREFIX + "/" + zip_filename
        utils.download_and_extract_archive(
            url,
            download_root=self.root,
            filename=zip_filename,
            md5=_ZIPS_MD5[self.zip_name_no_ext],
        )


data_augmentation_pipeline = solt.Stream(
    [
        slt.Rotate(p=0.5, angle_range=10.0),
        slt.Shear(p=0.5, range_x=0.3, range_y=0.3),
        slt.Scale(p=0.5, range_x=(0.8, 1.2), range_y=(0.8, 1.2)),
        slt.Translate(p=0.5, range_x=2, range_y=2),
    ]
)


def to_tensor(image: Image.Image) -> Tensor:
    return functional.to_tensor(image)


def to_augmented_tensor(image: Image.Image) -> Tensor:
    return data_augmentation_pipeline(
        solt.DataContainer(image, "I"),
        as_dict=False,
        normalize=False,
    )
