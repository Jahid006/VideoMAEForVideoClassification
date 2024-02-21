model_ckpt = "MCG-NJU/videomae-base"  # pre-trained model from which to fine-tune
batch_size = 2  # batch size for training and evaluation

from pathlib import Path
import os

os.environ["WANDB_DISABLED"] = "true"
from pathlib import Path
import datasets, os
import random


target_path = "charades_HF"
datasets.config.DOWNLOADED_DATASETS_PATH = Path(target_path)
dataset = datasets.load_dataset("HuggingFaceM4/charades")

label2id = {label: i for i, label in enumerate(list(range(157)))}
id2label = {i: label for label, i in label2id.items()}
dataset_root_path = "charades_HF/extracted/b1114f723459b513122220bfcf30b1b7423f65be17a11b936a8d29dbf1d83eba/Charades_v1"


def get_path_and_label(dataset, video_root, split="train"):
    paths = [
        os.path.join(video_root, video_info["video_id"] + ".mp4")
        for video_info in dataset[split]
    ]
    labels = [video_info["labels"] for video_info in dataset[split]]

    return paths, labels


train_path, train_labels = get_path_and_label(dataset, dataset_root_path)
test_path, test_labels = get_path_and_label(dataset, dataset_root_path, split="test")

zipped_train = list(zip(train_path, train_labels))

random.seed(37)
random.shuffle(zipped_train)

split_point = int(len(zipped_train) * 0.9)
zipped_train, zipped_eval = zipped_train[:split_point], zipped_train[split_point:]

train_path, train_labels = zip(*zipped_train)
eval_path, eval_labels = zip(*zipped_eval)


len(train_path), len(train_labels), len(eval_path), len(eval_labels), len(
    test_path
), len(test_labels)


from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
model = VideoMAEForVideoClassification.from_pretrained(
    model_ckpt, label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True
)

model.config.problem_type = "multi_label_classification"
model.config.num_labels = len(label2id)


import pytorchvideo.data

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)


import os

mean = image_processor.image_mean
std = image_processor.image_std
if "shortest_edge" in image_processor.size:
    height = width = image_processor.size["shortest_edge"]
else:
    height = image_processor.size["height"]
    width = image_processor.size["width"]
resize_to = (height, width)

num_frames_to_sample = model.config.num_frames
sample_rate = 4
fps = 30
clip_duration = num_frames_to_sample * sample_rate / fps

train_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(resize_to),
                    RandomHorizontalFlip(p=0.5),
                ]
            ),
        ),
    ]
)


# Validation and evaluation datasets' transformations.
val_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    Resize(resize_to),
                ]
            ),
        ),
    ]
)


import torch
from pytorchvideo_data import LabeledVideoDataset


def make_one_hot_label(label):
    labels = torch.zeros(len(id2label), dtype=torch.float32)
    labels[label] = 1.0
    return labels


# (List[Tuple[str, Optional[dict]]])
train_labeled_video_paths = [
    (v_path, {"label": make_one_hot_label(v_label)})
    for (v_path, v_label) in zip(train_path, train_labels)
]
eval_labeled_video_paths = [
    (v_path, {"label": make_one_hot_label(v_label)})
    for (v_path, v_label) in zip(eval_path, eval_labels)
]

train_dataset = LabeledVideoDataset(
    labeled_video_paths=train_labeled_video_paths,
    clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
    video_sampler=torch.utils.data.RandomSampler,
    transform=train_transform,
    decode_audio=False,
    decoder="pyav",
)

eval_dataset = LabeledVideoDataset(
    labeled_video_paths=eval_labeled_video_paths,
    clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
    video_sampler=torch.utils.data.RandomSampler,
    transform=val_transform,
    decode_audio=False,
    decoder="pyav",
)

len(train_dataset._labeled_videos), train_dataset._labeled_videos


from transformers import TrainingArguments, Trainer

model_name = model_ckpt.split("/")[-1]
new_model_name = f"{model_name}-finetuned-charades"
num_epochs = 1

args = TrainingArguments(
    new_model_name,
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    # push_to_hub=True,
    max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
)

import evaluate

metric = evaluate.load("accuracy")


# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions."""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


import torch


def collate_fn(examples):
    """The collation function to be used by `Trainer` to prepare data batches."""
    # permute to (num_frames, num_channels, height, width)
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.stack([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=image_processor,
    # compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

train_results = trainer.train()
trainer.save_model()
