# Report Summary: PlantDoc Visual Plant Disease Detection

## Objective
Build an end-to-end pipeline that detects plant leaves in real-world photos and classifies them as healthy or as a specific disease, then evaluate how far a lightweight modern detector can go on a small, noisy, in-the-wild dataset.

## Dataset
- PlantDoc (Roboflow 416×416 export): 2,567 internet images with bounding boxes. Training and test splits were provided, with no validation set.
- Images vary widely in background, lighting, scale and number of leaves, and some leaves are not annotated.
- After cleaning: 27 classes across 13 crop species. A 10% validation split was created from the training set.

## Data cleaning
- Repaired malformed annotation rows and verified that every image has annotations and every annotation has an image.
- Checked for duplicate images and rows, and normalised file names.
- Fixed a misspelled class name, corrected one mislabelled image, and removed two classes with only 2–3 images.

## Exploratory analysis
- Compared images per class with boxes per class, since many boxes in few images can make a model rely on context.
- Analysed box sizes, aspect ratios, very small boxes, and per-class brightness, contrast and sharpness.
- Built average RGB heatmaps, which revealed a small set of colour-inverted images.

## Preprocessing
- Detected and corrected colour-inverted images using channel statistics.
- Removed tiny bounding boxes and, where needed, images left without boxes.
- Filtered extreme brightness, contrast and blur outliers, and applied CLAHE only to low-contrast images.
- Offline augmentation (Albumentations), background-only images to reduce false positives, and conversion to YOLO label format.

## Model and training
- YOLO11n (about 2.6M parameters), fine-tuned from pretrained weights at 640 px for 40 epochs, batch 32.
- Optimiser chosen automatically by Ultralytics (AdamW), with a cosine learning-rate schedule and early stopping.
- Augmentations matched realistic capture conditions: horizontal flips, small rotations, translation, scale, colour jitter and mosaic. Shear, perspective, MixUp and copy-paste were disabled because they distort leaf shape or blend images.

## Handling incomplete labels
- **Run 2:** the baseline model mined likely missing boxes in the training set. Only confident, non-overlapping predictions of classes already present in the image were added.
- **Run 3:** likely unlabeled leaf regions were masked with the image's mean colour instead, protecting all ground-truth boxes.

## Results (test set: 238 images, 454 objects)
| Run | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|
| Run 1 (baseline) | 0.558 | 0.572 | 0.587 | 0.481 |
| Run 2 (mined labels) | 0.577 | 0.522 | 0.572 | 0.469 |
| Run 3 (cutout) | 0.529 | 0.584 | 0.558 | 0.449 |

## Conclusions and limitations
- The clean baseline gave the best overall accuracy. The label-correction strategies shifted the precision–recall balance but didn't raise mAP.
- Performance is strong for visually distinctive classes (strawberry, corn rust, peach) and weak for similar tomato diseases, where most errors are wrong-class predictions.
- The main limits are the small dataset size, class imbalance and inconsistent annotations. Larger models, a two-stage detector-plus-classifier design, and more data for the weakest classes are natural next steps.
