<img width="718" height="366" alt="image" src="https://github.com/user-attachments/assets/5ce43caa-b55b-4b4f-9a87-e0dee7790e26" /># Human Face Reconstruction Using PCA (Eigenfaces)

This project explores how computers can learn, simplify, and reconstruct human faces using a mathematical tool called Principal Component Analysis (PCA). The program trains on a baseline database of human portraits to extract primary facial blueprints—known as **Eigenfaces**—and uses those blueprints to recreate a completely new custom smartphone selfie.

---

## What is the Problem?

Digital images are made up of thousands or millions of individual pixels. For a computer, trying to analyze, recognize, or process faces directly using raw pixel data is incredibly difficult and inefficient for two main reasons:
1. **High Dimensionality:** Even a tiny 32x32 pixel grayscale image contains 1,024 individual pixel values that the computer must track simultaneously.
2. **Redundant Data:** Most pixels in a portrait are highly dependent on their neighbors. For example, if one pixel is part of a cheek or a forehead, the surrounding pixels are likely to show the exact same color intensity. Processing every single pixel independently creates massive computational waste.

---

## Project Objective

The goal of this project is to take a large collection of human face images, strip away the redundant pixel noise, and find the absolute minimum number of mathematical features needed to accurately describe a human face. 

Once the system learns these core features from the dataset, we test its understanding by introducing a completely unseen image—a personal smartphone selfie—and asking the computer to reconstruct the selfie using only the basic facial building blocks it discovered during training.

---

## What is PCA?

**Principal Component Analysis (PCA)** is a dimensionality reduction technique. In simple terms, it takes a massive dataset with hundreds of variables and condenses it down into a handful of core components without losing the important information.

When applied to human faces, PCA looks at how pixels change across different images. It figures out the lines, shapes, and lighting structures that vary the most from person to person (like the bridge of a nose, the outline of a jaw, or the depth of eye sockets). The dominant vectors it extracts are called **Eigenfaces**. Any human face can then be represented mathematically simply by mixing different proportions of these baseline Eigenfaces together.

---

## Our Approach: How We Solve It

Instead of forcing the computer to look at 1,024 individual pixels for every face, we use PCA to shift our perspective. 

We set a variance threshold of **95%**. This tells the mathematical engine to look at the entire dataset and figure out the minimum number of Eigenfaces required to capture 95% of the unique characteristics of human faces. By doing this, we compress the data drastically—dropping from 1,024 individual pixel dimensions down to just a small handful of principal coordinate dimensions.

---

## The Project Pipeline

The code operates step-by-step through a clean computer vision and machine learning pipeline:

### 1. Dataset Loading & Standardizing
The script reads a collection of grayscale human face images from the LFW (Labeled Faces in the Wild) dataset. To keep things uniform, every single image is automatically resized down to a standard 32x32 pixel grid.
<img width="722" height="190" alt="image" src="https://github.com/user-attachments/assets/8bdcba17-d1f4-4b10-89e9-42f82aa4959f" />

### 2. Data Flattening & Matrix Stacking
Each 32x32 pixel image grid is unrolled into a flat, one-dimensional list (vector) of 1,024 values. All of these individual vectors are stacked on top of one another to build a large 2D data matrix of shape $N \times 1,024$ (where $N$ is the total number of faces).

### 3. PCA Decomposition & Feature Extraction
PCA is applied directly to the 2D matrix. The algorithm calculates the eigenvectors and sorts them by importance. The top 5 mathematical components are isolated, reshaped back into 32x32 grids, and plotted visually as ghost-like "Eigenfaces" to show what the computer considers the most important facial traits.
<img width="746" height="177" alt="image" src="https://github.com/user-attachments/assets/899fa025-5486-4607-90ba-97015ea09819" />

### 4. Selfie Ingestion & Preprocessing
The program imports a custom personal smartphone selfie. Just like the training data, the selfie is converted to a clean single-channel grayscale format and scaled down to a matching 32x32 dimension.
<img width="509" height="531" alt="image" src="https://github.com/user-attachments/assets/4c91f288-14cb-45d5-a32b-f767a75e26aa" />
<img width="524" height="537" alt="image" src="https://github.com/user-attachments/assets/0736263e-e7d5-47ed-b7c5-efb563c4524b" />
<img width="550" height="539" alt="image" src="https://github.com/user-attachments/assets/8b868dce-f4d0-49e3-a632-bf09aae166c7" />

### 5. Projection & Linear Reconstruction
The processed selfie vector is projected directly into the low-dimensional Eigenface space. The computer calculates exactly how much weight to give to each baseline Eigenface to match your face. Finally, it performs a matrix multiplication to reconstruct the face from those weights and displays the final result side-by-side with your original selfie for a clear visual comparison.
<img width="739" height="423" alt="image" src="https://github.com/user-attachments/assets/c4f009e3-8497-44b9-8d96-9928dea1e7d2" />
---

## Visual Comparison Matrix

![Uploading image.png…]()



---

## Tech Stack & Core Libraries

* **Python 3:** The main programming environment.
* **NumPy:** Used to handle matrix manipulations, row stacking, and vector projections.
* **Pillow (PIL):** Used to open, resize, and convert your custom smartphone selfie to grayscale.
* **Scikit-Learn:** Provides the PCA algorithm and dataset utilities.
* **Matplotlib:** Used to display the eigenfaces and render the final side-by-side reconstruction plots.

---

## Quick Setup & Execution Guide

### Prerequisites
Make sure your custom smartphone selfie image file is saved inside your workspace directory, and install the required Python packages:

```bash
pip install numpy pillow scikit-learn matplotlib
