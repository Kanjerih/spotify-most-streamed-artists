# Spotify Most-Watched Artist Classification

## Project Overview

This project uses machine learning to investigate whether an artist's characteristics can be used to predict whether they belong to the top 20% of artists based on total streams.

The project uses a dataset of 500 Spotify artists and follows an end-to-end machine learning workflow, including data exploration, preprocessing, feature engineering, categorical encoding, classification, and model evaluation.

---

## Objective

The main question addressed in this project is:

> **Can an artist's characteristics predict whether they are a high-stream artist?**

Artists were classified as **High Stream Artists** if their total streams were at or above the 80th percentile of the dataset.

---

## Dataset

The dataset contains 500 artists and includes:

- Artist Name
- Sex
- Country of Origin
- Primary Language
- Primary Genre
- Artist Type
- Debut Year
- Total Streams
- Lead Streams
- Feature Streams
- Solo Streams
- % of Solo Streams
- Collaborative Streams
- % of Collaborative Streams

The dataset contained **500 rows and 14 columns**, with no missing values.

---

## Data Processing

### Data Inspection

The dataset was inspected for:

- Missing values
- Data types
- Categorical distributions
- Numerical distributions
- Skewness
- Correlations

`Debut Year` was retained as an integer because the dataset contained only the year, with values such as `2000`, `2005`, and `2015`.

### Numerical Analysis

The streaming variables were examined for skewness and correlations.

Several streaming variables were highly skewed, so log transformations were applied during exploratory analysis.

Correlation analysis showed, for example, a correlation of approximately **0.562** between Total Streams and Lead Streams.

These streaming variables were used for exploratory analysis but were not included as predictive features in the final classification model.

---

## Target Variable

The target variable `High Stream Artist` was created using the 80th percentile of Total Streams.

```python
threshold = spotify_df['Total Streams (in millions)'].quantile(0.8)

spotify_df['High Stream Artist'] = (
    spotify_df['Total Streams (in millions)'] >= threshold
).astype(int)
