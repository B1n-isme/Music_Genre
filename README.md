# Music Genre Classification and Recommendation System

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR4nckY8bOI0_m8q-sVQoYnwd1JcwhB7JWF9g&s" alt="GTZAN Dataset" style="width:100%;"/>

This project focuses on classifying music genres using the **GTZAN dataset** and constructing a recommendation system using a **voting ensemble** of multiple similarity-based methods. The project is divided into several components covering signal preprocessing, data visualization, classification models, and the recommendation system.

## Table of Contents

- [Dataset](#dataset)
- [Project Components](#project-components)
  - [Signal Preprocessing and Feature Extraction](#signal-preprocessing-and-feature-extraction)
  - [Data Visualization](#data-visualization)
  - [Music Genre Classification](#music-genre-classification)
  - [Music Recommendation System](#music-recommendation-system)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)

---

## Dataset

The project utilizes the **GTZAN dataset** for music genre classification. The dataset consists of 10 genres, with 100 tracks per genre, each 30 seconds long. The genres include:

- Blues
- Classical
- Country
- Disco
- Hip-hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

### Download the Dataset
You can download the GTZAN dataset from this  [Kaggle Link](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).

---

## Project Components

### Signal Preprocessing and Feature Extraction

The first component of the project is an exploration of the **preprocessing** of audio signals and the extraction of 12 key features for each song. These features are crucial for understanding the underlying characteristics of the audio signals and include:

1. **Waveform**
2. **Sampling Rate**
3. **Spectrogram**
4. **Mel-Spectrogram**
5. **RMS Energy (RMS-E)**
6. **Zero-Crossing Rate**
7. **Spectral Roll-off**
8. **Spectral Centroid**
9. **Spectral Bandwidth**
10. **Chroma**
11. **Harmonic/Percussive Source Separation (HPSS)**
12. **Mel-frequency Cepstral Coefficients (MFCC)**

You can find the detailed **signal preprocessing** and **feature extraction** process, along with visualizations of each feature, in the [1_preprocess_signal.ipynb](./1_preprocess_signal.ipynb) notebook.

### Data Visualization

This component focuses on visualizing the processed features of the music data. The following plots are used for exploration:

- **Heatmap**: Provides a correlation matrix of the features.
- **Boxplot**: Displays the distribution of features across different genres.
- **PCA Scatterplot**: Visualizes the music data reduced to two dimensions using **Principal Component Analysis (PCA)**.

You can find the visualizations in the [2_music_genre_visualization.ipynb](./2_music_genre_visualization.ipynb) notebook.

### Music Genre Classification

This component includes the implementation of various classification models to predict the genre of a music track. Models include:

- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Custom Multi-Layer Perceptron (MLP)**

You can find the classification models and their evaluation in the [3_music_genre_classification.ipynb](./3_music_genre_classification.ipynb) notebook.

### Music Recommendation System

A custom music recommendation system is built using an ensemble of **four methods**:

1. **Cosine Similarity**
2. **Truncated SVD**
3. **Autoencoder**
4. **Centroid Distance-Based Similarity**

These methods are combined in a **voting ensemble** to recommend songs based on similarity within the dataset. The recommendation system can suggest similar songs based on the content and features of a given track.

The recommendation system is implemented in the [4_music_recommend_sys.ipynb](./4_music_recommend_sys.ipynb) notebook.

---

## Installation

To run the project, ensure you have the following libraries installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn librosa keras
```

---

## Usage
1. **Signal Preprocessing and Feature Extraction**: Run [2_music_genre_visualization.ipynb](./2_music_genre_visualization.ipynb) to preprocess the audio signals and extract the features.
2. **Data Visualization**: Run [2_music_genre_visualization.ipynb](./2_music_genre_visualization.ipynb) to explore the data visually with heatmaps, boxplots, and PCA scatterplots.
3. **Music Genre Classification**: Run [3_music_genre_classification.ipynb](./3_music_genre_classification.ipynb) to train and evaluate the genre classification models.
4. **Music Recommendation System**: Run [4_music_recommend_sys.ipynb](./4_music_recommend_sys.ipynb) to construct a recommendation system using the ensemble of similarity-based methods.

---

## References
1. GTZAN Dataset: Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals. IEEE Transactions on Speech and Audio Processing.
2. Librosa: A Python package for music and audio analysis.
3. Scikit-learn: Machine learning library for Python.
4. Keras: Deep learning library for Python.


### Explanation:

1. **Dataset Section**: Provides details on the GTZAN dataset, including genres and a link to download it.
2. **Project Components**: Describes the structure of the project and the functionality in each Jupyter notebook:
   - **Signal Preprocessing and Feature Extraction**: Describes the features and links to the signal processing notebook.
   - **Data Visualization**: Describes visualizations like heatmaps, boxplots, and PCA scatterplots.
   - **Music Genre Classification**: Describes the classification models implemented.
   - **Music Recommendation System**: Details the recommendation system using a voting ensemble of different methods.
3. **Installation Section**: Includes a list of required libraries.
4. **Usage Section**: Explains how to run each part of the project.
5. **References**: Lists relevant sources related to the dataset and tools.
