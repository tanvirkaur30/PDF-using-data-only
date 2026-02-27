# Learning Probability Density Function using GAN

## Using NO₂ Air Quality Data

---

## 1. Introduction

Estimating the probability density function (PDF) of a random variable is a fundamental problem in statistics and machine learning. Traditional approaches assume a known parametric form (e.g., Gaussian), which often fails for real-world data.

This project explores a **data-driven approach** using a **Generative Adversarial Network (GAN)** to learn an unknown PDF directly from samples, without making any assumptions about its analytical form.

---

## 2. Dataset Description

- Dataset: India Air Quality Dataset  
- Feature Used: NO₂ concentration (`no2`)  
- Source: https://www.kaggle.com/datasets/shrutibhargava94/india-air-quality-data  

The dataset contains real-world pollution measurements collected across various locations, resulting in a **non-uniform and noisy distribution**.

---

## 3. Mathematical Transformation

The original variable \( x \) is transformed into a new variable \( z \) using:

\[
z = x + a_r \cdot \sin(b_r \cdot x)
\]

Where:
- Roll Number \( r = 102303389 \)
- \( a_r = 3.0 \)
- \( b_r = 1.5 \)

### Interpretation

- The sine component introduces **periodic non-linearity**
- This results in:
  - Distortion of the original distribution  
  - Creation of local fluctuations and irregular density regions  
- The transformed variable \( z \) becomes **harder to model using standard distributions**

---

## 4. Data Preprocessing

The following preprocessing steps were applied:

1. **Missing Value Removal**
   - Used `dropna()` to remove incomplete entries

2. **Feature Extraction**
   - Selected only the `no2` column

3. **Normalization**
   - Standardized data using:
     \[
     x' = \frac{x - \mu}{\sigma}
     \]
   - This ensures:
     - Faster convergence  
     - Stable GAN training  

---

## 5. GAN Architecture

A basic GAN architecture is implemented using PyTorch.

### 5.1 Generator Network

- Input: 1D Gaussian noise \( z \sim N(0,1) \)
- Architecture:
  - Linear (1 → 16)
  - ReLU activation
  - Linear (16 → 1)

**Role:**  
Transforms random noise into synthetic samples that mimic the real data distribution.

---

### 5.2 Discriminator Network

- Input: Real or generated samples
- Architecture:
  - Linear (1 → 16)
  - ReLU activation
  - Linear (16 → 1)
  - Sigmoid activation

**Role:**  
Classifies whether a sample is real (from dataset) or fake (from generator).

---

## 6. Training Methodology

The GAN is trained using an adversarial process:

### Step 1: Train Discriminator
- Maximize:
  - Probability of classifying real samples correctly  
  - Probability of identifying fake samples  

### Step 2: Train Generator
- Minimize:
  - Discriminator’s ability to detect fake samples  

---

### Training Configuration

- Loss Function: Binary Cross Entropy (BCELoss)
- Optimizer: Adam
- Learning Rate: 0.001
- Epochs: 7000
- Batch Size: 64

---

### Training Behavior (Observed)

- Discriminator Loss ≈ 1.38  
- Generator Loss ≈ 0.68–0.73  

This indicates:
- Discriminator is **not strongly distinguishing** real vs fake  
- Generator is learning **approximate distribution**, but not precise details  

---

## 7. PDF Approximation

After training:

1. Generated **7000 synthetic samples** using the generator  
2. Estimated PDF using **histogram-based density estimation**  

Both real and generated distributions were plotted for comparison.

---

## 8. Results and Analysis

### Visual Comparison

- The generated distribution follows the general trend of the real data  
- Overlap is observed in the central region  

---

### Detailed Observations

#### Mode Coverage
- GAN captures the **overall shape** of the distribution  
- However, smaller peaks and fluctuations are not well learned  

#### Distribution Alignment
- Slight shift between real and generated distributions  
- Spread (variance) is not perfectly matched  

#### Tail Behavior
- GAN struggles to capture extreme values (tails)  
- Generated samples are more concentrated  

---

## 9. Limitations

- Shallow architecture limits representational power  
- Lack of stabilization techniques affects training quality  
- Histogram-based PDF is less accurate than KDE  
- Possible underfitting due to weak discriminator  

---

## 10. Improvements and Future Work

To improve performance:

- Use **LeakyReLU** instead of ReLU  
- Add more layers and neurons  
- Apply **label smoothing**  
- Introduce noise to discriminator input  
- Replace histogram with **Kernel Density Estimation (KDE)**  
- Use advanced GAN variants:
  - Wasserstein GAN (WGAN)
  - Conditional GAN (CGAN)

---

## 11. Conclusion

This project demonstrates that GANs can learn an approximate probability density function directly from data without assuming any parametric form.

While the model successfully captures the general distribution, it struggles with fine details due to architectural simplicity and training limitations.

This highlights both the **power and challenges of GAN-based distribution learning**.

---

## 12. Output

<img width="726" height="535" alt="image" src="https://github.com/user-attachments/assets/c8915419-b243-4335-96bd-f511717907a4" />

