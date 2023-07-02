import random
from os import path

import solt
import torch
from PIL import Image
from solt import transforms as slt
from torch.utils.data import IterableDataset
from torchvision.datasets import utils
from torchvision.transforms import functional

_SUB_ROOT_DIR = "omniglot-py"
_IMAGES_EVALUATION: str = "images_evaluation"
_IMAGES_BACKGROUND: str = "images_background"
_DOWNLOAD_URL_PREFIX: str = (
    "https://raw.githubusercontent.com/brendenlake/omniglot/master/python"
)
_ZIPS_MD5: dict[str, str] = {
    _IMAGES_BACKGROUND: "68d2efa1b9178cc56df9314c21c6e718",
    _IMAGES_EVALUATION: "6b91aef0f799c5bb55b94e3f2daec811",
}
_NUM_CHARS_PER_ALPHA: int = 15  # number of characters to take per alphabet
_SAME_CLASS_TARGET = torch.ones(size=())  # 0-dimensional i.e. just a scalar 1.0
_DIFF_CLASS_TARGET = torch.zeros(size=())  # 0-dimensional i.e. just a scalar 0.0


class OmniglotForVerificationTask(IterableDataset):
    data: list[list[list[Image]]]

    def __init__(
        self,
        root: str,
        train: bool = True,
        n_samples: int = 30_000,
        augment: bool = False,
        download: bool = False,
    ) -> None:
        if isinstance(root, str):
            root = path.expanduser(root)
        self.root = path.join(root, _SUB_ROOT_DIR)
        self.train = train
        assert (
            n_samples > 0 and n_samples % 2 == 0
        ), "n_samples must be a positive even number"
        self.n_sample_pairs = n_samples
        if augment:
            assert train, "augment can only be True for training set"
        self.augment = augment
        self.zip_name_no_ext = _IMAGES_BACKGROUND if train else _IMAGES_EVALUATION
        self.n_drawers = 12 if train else 16
        if download:
            self.download_and_prepare()
        else:
            self.prepare_data()

    def download_and_prepare(self):
        self._download()
        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it"
            )
        self.prepare_data()

    def prepare_data(self):
        self.data = []
        root = path.join(self.root, self.zip_name_no_ext)
        for alphabet in utils.list_dir(root):
            characters = utils.list_dir(path.join(root, alphabet))
            if len(characters) < _NUM_CHARS_PER_ALPHA:
                continue
            self.data.append(
                [
                    [
                        Image.open(
                            path.join(root, alphabet, character, drawing), mode="r"
                        ).convert("L")
                        for drawing in utils.list_files(
                            root=path.join(root, alphabet, character),
                            suffix=".png",
                        )[: self.n_drawers]
                    ]
                    for character in random.sample(characters, _NUM_CHARS_PER_ALPHA)
                ]
            )

    def __iter__(self):
        n_alphas = len(self.data)
        characters_range = range(_NUM_CHARS_PER_ALPHA)
        drawers_range = range(self.n_drawers)
        times_to_augment = 8 if self.augment else 0

        for _ in range(self.n_sample_pairs // 2):
            a = random.randrange(n_alphas)
            c1, c2 = random.sample(characters_range, 2)
            d1, d2 = random.sample(drawers_range, 2)

            anchor = self.data[a][c1][d1]
            positive = self.data[a][c1][d2]
            anchor_tensor = to_tensor(anchor)
            yield anchor_tensor, to_tensor(positive), _SAME_CLASS_TARGET

            negative = self.data[a][c2][d2]
            yield anchor_tensor, to_tensor(negative), _DIFF_CLASS_TARGET

            for _ in range(times_to_augment):
                anchor_tensor = to_augmented_tensor(anchor)
                yield anchor_tensor, to_augmented_tensor(positive), _SAME_CLASS_TARGET
                yield anchor_tensor, to_augmented_tensor(negative), _DIFF_CLASS_TARGET

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
            url, self.root, filename=zip_filename, md5=_ZIPS_MD5[self.zip_name_no_ext]
        )


augmentation_pipeline = solt.Stream(
    [
        slt.Rotate(p=0.5, angle_range=10.0),
        slt.Shear(p=0.5, range_x=0.3, range_y=0.3),
        slt.Scale(p=0.5, range_x=(0.8, 1.2), range_y=(0.8, 1.2)),
        slt.Translate(p=0.5, range_x=2, range_y=2),
    ]
)


def to_tensor(image: Image) -> torch.Tensor:
    return functional.to_tensor(image)


def to_augmented_tensor(image: Image) -> torch.Tensor:
    return augmentation_pipeline(
        solt.DataContainer(image, "I"), as_dict=False, normalize=False
    )
