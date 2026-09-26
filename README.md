# SRGAN

## File Structure

```text
srgan/
├── dataset/
├── dataset_SRGAN/
│
├── srgan/
│   ├── train/
│   └── val/
│
├── utils/
│   └── model.py
│
├── 01_SplitData.ipynb
├── 02_TrainSRGAN.ipynb
├── 03_UseSRGAN.ipynb
├── 04_EvaluationSRGAN.ipynb
│
├── result/
│   ├── epoch_images/
│   ├── model.pth
│   ├── metric_graph/
│   └── training_log/
│
└── README.md
```

### File Description

| File / Folder              | Description                                                                          |
| -------------------------- | ------------------------------------------------------------------------------------ |
| `dataset/`                 | Stores the original image dataset used for training.                                 |
| `dataset_SRGAN/`           | Stores the SRGAN image dataset that is generated from the original dataset.          |
| `srgan`                    | Stores the dataset used to train the SRGAN model.                                    |
| `utils`                    | Stores the SRGAN Generator. (model.py)                                               |
| `01_SplitData.ipynb`       | Splits the dataset into training and validation sets.                                |
| `02_TrainSRGAN.ipynb`      | Trains the SRGAN model using the prepared dataset.                                   |
| `03_UseSRGAN.ipynb`        | Uses the trained SRGAN model to generate Super-Resolution (SR) images.               |
| `04_EvaluationSRGAN.ipynb` | Evaluates the SRGAN results using image quality metrics such as MSE, PSNR, and SSIM. |
| `result/`                  | Stores training results and generated outputs.                                       |
| `epoch_images/`            | Stores generated images during training for monitoring model progress.               |
| `model.pth`                | The trained SRGAN model weights.                                                     |
| `metric_graph/`            | Stores graphs of training metrics.                                                   |
| `training_log/`            | Stores training logs and related information.                                        |

---

# How to Run

Follow the steps below in order to train and evaluate the SRGAN model.

### 1. Clone Repository and Prepare CUDA

First, clone the repository:

```bash
git clone https://github.com/kulthapon/srgan.git
cd srgan
```

Make sure that **CUDA** is installed and properly configured on your system before running the notebooks.

You can verify that CUDA is available through PyTorch:

```python
import torch

print(torch.cuda.is_available())
```

The output should be:

```text
True
```


### 2. Prepare the Dataset

Place the dataset that you want to use for SRGAN training inside the `dataset/` folder.

```text
srgan/
└── dataset/
    ├── image_001.jpg
    ├── image_002.jpg
    ├── image_003.jpg
    └── ...
```

Make sure the images have the appropriate resolution before proceeding.

---

### 3. Run `01_SplitData.ipynb`

Run all cells in **`01_SplitData.ipynb`** to prepare and split the dataset.

The processed dataset will be stored in:

```text
srgan/
```

---

### 4. Run `02_TrainSRGAN.ipynb`

Run **`02_TrainSRGAN.ipynb`** to train the SRGAN model.

The training results will be saved in the `result/` folder, including:

* Generated images during training
* Training metrics
* Training logs
* Trained model weights

The trained model will be saved as:

```text
result/model.pth
```

---

### 5. Run `03_UseSRGAN.ipynb`

After training is complete, run **`03_UseSRGAN.ipynb`** to load the trained model and generate Super-Resolution (SR) images.

Make sure the path to `model.pth` is correctly configured before running the notebook.

The generated images will be saved as:

```text
dataset_SRGAN/
```

---

### 6. Run `04_EvaluationSRGAN.ipynb`

Finally, run **`04_EvaluationSRGAN.ipynb`** to evaluate the generated SR images.

The evaluation includes image quality metrics such as:

* **MSE (Mean Squared Error)**
* **PSNR (Peak Signal-to-Noise Ratio)**
* **SSIM (Structural Similarity Index)**
