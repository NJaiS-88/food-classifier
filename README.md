# Food Classification — Burger, Pizza & Soft Drink

A Convolutional Neural Network (CNN) trained to classify food images into three categories: **Pizza**, **Burger**, and **Soft Drink**. Built with TensorFlow/Keras on the [Kaggle Food Classification Dataset](https://www.kaggle.com/datasets/manishkc06/food-classification-burger-pizza-coke).

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Takeaways](#key-takeaways)

---

## Overview

This project demonstrates a custom CNN pipeline for 3-class food image classification. The workflow covers data ingestion from Kaggle, class-wise folder organisation, data augmentation, model training with class weighting, and single-image inference — all in Google Colab.

---

## Dataset

| Split      | Images | Classes               |
|------------|--------|-----------------------|
| Training   | 4,320  | Pizza, Softdrinks, Burgers |
| Validation | 1,080  | Pizza, Softdrinks, Burgers |
| **Total**  | **5,400** | 1,800 per class    |

**Preprocessing:**
- Images resized to `128 × 128`
- Pixel values normalised to `[0, 1]`
- Training augmentation: shear (0.2), zoom (0.2), horizontal flip
- 80/20 train-validation split via `ImageDataGenerator`

---

## Model Architecture

A sequential CNN with three convolutional blocks followed by fully-connected layers.

```
Input (128 × 128 × 3)
  └── Conv2D(32, 3×3, relu) → BatchNorm → MaxPool(2×2)
  └── Conv2D(64, 3×3, relu) → BatchNorm → MaxPool(2×2)
  └── Conv2D(128, 3×3, relu) → BatchNorm → MaxPool(2×2)
  └── Flatten
  └── Dense(128, relu)
  └── Dense(64, relu)
  └── Dense(3, softmax)
```

**Total parameters:** 3,313,987 (~12.64 MB)

---

## Training

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_gen,
    steps_per_epoch=len(train_gen),
    epochs=10,
    validation_data=val_gen,
    validation_steps=len(val_gen),
    class_weight={0: 2.0, 1: 1.0, 2: 2.0}   # Pizza=2, Softdrinks=1, Burgers=2
)
```

| Hyperparameter   | Value                    |
|------------------|--------------------------|
| Optimizer        | Adam                     |
| Loss             | Categorical Crossentropy |
| Epochs           | 10                       |
| Batch size       | 32                       |
| Validation split | 20%                      |

---

## Results

| Epoch | Train Accuracy | Val Accuracy | Val Loss |
|-------|---------------|-------------|----------|
| 1     | 69.35%        | 33.43%      | 2.999    |
| 2     | 75.53%        | 57.50%      | 1.273    |
| 3     | 78.89%        | 66.94%      | 1.030    |
| 4     | 81.30%        | 77.22%      | 0.598    |
| 5     | 83.03%        | 83.33%      | 0.447    |
| 6     | 85.88%        | 82.13%      | 0.453    |
| 7     | 85.72%        | 79.35%      | 0.604    |
| 8     | 86.11%        | 75.46%      | 0.738    |
| 9     | 88.08%        | 77.13%      | 0.672    |
| **10**| **90.67%**    | **84.07%**  | **0.449**|

**Final validation accuracy: 84.07%** | **Final validation loss: 0.4489**

> Applying class weights `{Pizza: 2, Softdrinks: 1, Burgers: 2}` improved validation accuracy by ~5–7% compared to training without them, by encouraging the model to pay more attention to harder-to-distinguish classes.

---

## Usage

### Prerequisites

```bash
pip install tensorflow keras numpy matplotlib kagglehub
```

### Run inference on a single image

```python
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Class mapping
CLASS_NAMES = {0: "Pizza", 1: "Softdrinks", 2: "Burgers"}

model = load_model("my_model.keras")

img = image.load_img("your_image.jpg", target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)[0]

print(f"Predicted: {CLASS_NAMES[predicted_class]}")
```

### Save and load the model

```python
model.save("my_model.keras")          # Save

from tensorflow.keras.models import load_model
model = load_model("my_model.keras")  # Load
```

---

## Project Structure

```
food-classification/
├── food-classification-burger-pizza-coke/
│   ├── train/train/          # Raw training images
│   └── Training_set_food.csv # Filename → label mapping
├── sorted_data/
│   ├── Pizza/                # 1,800 images
│   ├── Softdrinks/           # 1,800 images
│   └── burgers/              # 1,800 images
├── my_model.keras            # Saved trained model
└── food_classification.ipynb # Main Colab notebook
```

---

## Key Takeaways

- **Class weights matter.** Assigning higher weight to Pizza and Burgers boosted validation accuracy from ~77% to 84%, compensating for the model's initial bias towards Softdrinks.
- **Batch normalisation** after each conv layer stabilised training and helped the model converge faster.
- **Data augmentation** (shear, zoom, flip) reduced overfitting on the training set and improved generalisation.
- The model achieved consistent predictions across varied real-world test images (screenshots from different sources), demonstrating reasonable robustness.

---

## License

Dataset license: see [Kaggle dataset page](https://www.kaggle.com/datasets/manishkc06/food-classification-burger-pizza-coke).
