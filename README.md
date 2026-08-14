# OK-NOK Image Classifier

A simple image classification project that uses a **ResNet-18** model to classify images as **OK** or **NOK**.

The idea is straightforward: give the model images of correct and incorrect parts, train it, and then use it to check new images.

## Features

- ResNet-18 with transfer learning
- OK / NOK image classification
- Data augmentation
- Stratified K-Fold cross-validation
- Early stopping
- Confidence score
- OpenCV result visualization

## How it works

The project has four main steps:

1. Load the images and assign labels based on their filenames.
2. Train a ResNet-18 model using the images.
3. Validate the model using K-Fold cross-validation.
4. Use the trained model to classify a new image and save the result.

Example:

```text
Image
  ↓
ResNet-18
  ↓
OK / NOK
  ↓
Confidence score
  ↓
Result image
```

## Model

The project uses **ResNet-18**, a relatively small and fast CNN that works well with transfer learning.

Most of the pre-trained network is kept frozen, while the last layer (`layer4`) and the custom classifier are trained for this specific task.

The classifier is:

```text
Dropout
   ↓
Linear (512 → 128)
   ↓
ReLU
   ↓
Dropout
   ↓
Linear (128 → 2)
```

## Project Structure

```text
OK-NOK-image-classifier/
├── data/
│   ├── OK_01.jpg
│   ├── OK_02.png
│   ├── NOK_01.jpg
│   └── NOK_02.png
├── train_custom.py
├── requirements.txt
└── README.md
```

## Installation

You need:

- Python 3.8+
- PyTorch
- OpenCV
- The packages listed in `requirements.txt`

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
.\venv\Scripts\activate
```

**Linux/macOS**
```bash
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Preparing the Data

Put the training images inside the `data/` folder.

The filename determines the class:

```text
OK_part1.png       → OK
OK_sample.jpg      → OK

NOK_defect1.jpg    → NOK
NOK_wrong_side.png → NOK
```

## Running the Project

Run:

```bash
python train_custom.py
```

The script will:

1. Load the images from `data/`.
2. Split the data for training and validation.
3. Train the model.
4. Save the best model checkpoint.
5. Classify a sample image.
6. Create an annotated result image.

The output will show something similar to:

```text
OK: Correct Side (95%)
```

or

```text
NOK: WRONG SIDE (91%)
```

## Main Settings

| Setting | Value |
|---|---|
| Model | ResNet-18 |
| Image size | 224 × 224 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | Up to 8 |
| Maximum epochs | 15 |
| Early stopping | 5 epochs |
| Loss | CrossEntropyLoss |

## Data Augmentation

During training, images can be slightly rotated, cropped, and adjusted for brightness and contrast.

This helps the model handle small changes in camera position and lighting.

## Result Visualization

OpenCV is used to add a simple result to the image:

- Green border → **OK**
- Red border → **NOK**
- Confidence percentage is displayed on the image.

## License

This project is available under the MIT License.
