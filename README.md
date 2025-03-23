# Iris Flower Classification Project

## Overview
This project implements a **K-Nearest Neighbors (KNN) classification model** to classify Iris flower species based on the well-known **Iris dataset**. The dataset consists of measurements of **sepal length, sepal width, petal length, and petal width**, with the goal of predicting the corresponding flower species.

## Dataset
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)
- Features:
  - **Sepal Length**
  - **Sepal Width**
  - **Petal Length**
  - **Petal Width**
- Target: **Iris species (Setosa, Versicolor, Virginica)**

## Libraries Used
- **pandas** – Data manipulation and analysis
- **numpy** – Numerical computations
- **matplotlib & seaborn** – Data visualization
- **scikit-learn** – Machine learning (train-test split, KNN classifier, evaluation metrics)

## Project Steps
1. **Load and Explore the Dataset**
2. **Visualize Data Distribution** (using pair plots)
3. **Preprocess Data** (splitting into training and testing sets)
4. **Train KNN Classifier** (with `n_neighbors=3`)
5. **Evaluate Model Performance** (accuracy score & classification report)

## Results
The trained **KNN model achieved high accuracy** in classifying Iris species. Below are key performance metrics:
- **Accuracy:** Displayed after model evaluation
- **Classification Report:** Precision, recall, and F1-score breakdown

## How to Run the Project
1. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
2. Run the script in a Jupyter Notebook or Google Colab.

## Visualization
A **pair plot** of the dataset shows clear separation between classes:
```python
sns.pairplot(iris_data, hue="Class")
plt.show()
```

## Future Improvements
- Experiment with different machine learning models (e.g., SVM, Decision Trees)
- Tune hyperparameters of KNN for better performance
- Deploy the model as a web app using Flask or Streamlit

## Author
Developed by **Kerolos Amgad **

## License
This project is open-source and available under the **MIT License**.

