<div align="center">

<br>
<h1>Tuning of Diffusion Models for <br> Enhancing Synthetic Data Usage Ratio</h1>

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub last commit](https://img.shields.io/github/last-commit/JohnYechanJo/Novo-Nordisk_Synthetic-Image-Usage)](https://github.com/JohnYechanJo/Novo-Nordisk_Synthetic-Image-Usage/commits/main)
[![GitHub issues](https://img.shields.io/github/issues/JohnYechanJo/Novo-Nordisk_Synthetic-Image-Usage)](https://github.com/JohnYechanJo/Novo-Nordisk_Synthetic-Image-Usage/issues)
<br>
## Contributors

[![GitHub Contributors](https://img.shields.io/github/contributors-anon/JohnYechanJo/Novo-Nordisk_Synthetic-Image-Usage)](https://github.com/JohnYechanJo/Novo-Nordisk_Synthetic-Image-Usage/graphs/contributors)

<table>
  <tr>
<td align="center"><a href="https://github.com/JohnYechanJo"><img src="https://avatars.githubusercontent.com/u/131790222?v=4" width="100px;" alt=""/><br /><sub><b>John Yechan Jo</b></sub></a><br /></td>
<td align="center"><a href="https://github.com/Kaaaaaaaaaaaai"><img src="https://avatars.githubusercontent.com/u/117432135?v=4" width="100px;" alt=""/><br /><sub><b>Kazuma Itabashi</b></sub></a><br /></td>
<td align="center"><a href="https://github.com/allergic-garlic"><img src="https://avatars.githubusercontent.com/u/116773411?v=4" width="100px;" alt=""/><br /><sub><b>Yiran Qi</b></sub></a><br /></td>
<td align="center"><a href="https://github.com/joelleoqiyi"><img src="https://avatars.githubusercontent.com/u/20594310?v=4" width="100px;" alt=""/><br /><sub><b>Joel Leo</b></sub></a><br /></td>
  </tr>
</table>

</div>

---

## Motivation
Choroidal Neovascularization (CNV) is a critical condition characterized by abnormal blood vessel growth under the retina, diagnosed using Optical Coherence Tomography (OCT) scans. Our project investigates the use of diffusion models to generate high-quality synthetic OCT images, augmenting limited datasets to improve binary classification of CNV vs. Normal images. We aim to determine the optimal synthetic-to-real data ratio (\( R \)) for robust model performance, addressing challenges in medical imaging.

---

## Features
- **Synthetic CNV Image Generation**: Fine-tuned a distilled Stable Diffusion model to produce realistic OCT scans for CNV.
- **Binary Classification**: Developed a CNN classifier to distinguish CNV from Normal OCT images, tested with varying synthetic data ratios.
- **Dataset Augmentation**: Leveraged the OCT2017 dataset (83,484 training images) and generated 640 synthetic CNV images.
- **Performance Optimization**: Achieved peak classifier performance at \( R = 50\% \) (50% synthetic, 50% real).
- **Image Quality Analysis**: Visualized well-generated vs. poorly-generated synthetic images to evaluate feature fidelity.

---

## Technical Details

### Dataset
- **OCT2017 Dataset**:
  - **Training Set**: 83,484 images (CNV: 37,205, Normal: 26,315).
  - **Validation Set**: 32 images (CNV: 8, Normal: 8).
  - **Test Set**: 968 images (CNV: 242, Normal: 242).
- Due to GPU constraints, a subset was used, augmented with 640 synthetic CNV images generated via a diffusion model.

### Model Architecture
- **Diffusion Model**:
  - Fine-tuned a nota-ai distilled Stable Diffusion model on 30,000 CNV images for 30 epochs (batch size: 8).
  - Preprocessed images to 512x512 resolution with normalized intensities.
  - Trained UNet with frozen VAE and tokenizer, using a learning rate of 1e-4 and up to 250 denoising steps.
  - Applied a guidance scale of ~12.5 to enhance CNV-specific features.
- **Classifier**:
  - Simple CNN with two convolutional layers, max pooling, and a fully connected layer.
  - Mixed real and synthetic CNV images at ratio \( R \) in training batches.

### Key Findings
- **Optimal Ratio**: Best performance at \( R = 50\% \) (Accuracy: 0.9090, F1 Score: 0.8999, Precision: 0.9010, Recall: 0.9090).
- **Overfitting at High Ratios**: Accuracy dropped significantly at \( R = 100\% \) (0.6813) due to synthetic image artifacts (e.g., smooth textures, contrast mismatches).
- **Image Quality**:
  - Well-generated images showed high-reflectance areas typical of CNV.
  - Poorly-generated images exhibited random patterns, failing to capture CNV features.

### Quality Assurance
- **Metrics**:
  - Evaluated Accuracy, F1 Score, Precision, and Recall across \( R \) from 10% to 90%.
  - Proposed Fréchet Inception Distance (FID) for future real-vs-synthetic quality assessment.
- **Challenges**:
  - Synthetic images sometimes introduced noise, reducing generalization at high \( R \).
  - Imbalanced dataset (e.g., fewer Drusen samples) highlighted the need for balanced augmentation.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/JohnYechanJo/Novo-Nordisk_Synthetic-Image-Usage
