from pathlib import Path

from fastai.vision.all import (
    CategoryBlock,
    DataBlock,
    ImageBlock,
    PILImage,
    RandomSplitter,
    Resize,
    error_rate,
    get_image_files,
    parent_label,
    resnet18,
    vision_learner,
)

PARENT_PATH = Path(__file__).parent
CAMELS_DATA_PATH = Path(PARENT_PATH, "data/camels")
MODEL_EXPORT_PATH = Path(PARENT_PATH, "camels_model")


def get_model():
    data_block = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(192, method="squish")],
    ).dataloaders(CAMELS_DATA_PATH, bs=32)
    learn = vision_learner(data_block, resnet18, metrics=error_rate)
    learn.fine_tune(3)
    return learn


if __name__ == "__main__":
    learn = get_model()
    learn.path = MODEL_EXPORT_PATH
    learn.export()
