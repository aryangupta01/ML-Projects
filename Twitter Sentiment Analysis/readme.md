# Covid Tweets Sentiment Analysis
This project implements a Twitter Sentiment Analysis system focused on detecting help-seeking tweets related to oxygen during the COVID-19 pandemic. Due to a small percentage of labeled data, the project leverages a semi-supervised learning approach and explores various machine learning models, including the advanced BERT transformer model.

## Project Overview

The primary objective of this project is to classify tweets into three categories: help-seeking, non-help-seeking, or neutral. The dataset consists of approximately 70,000 tweets, with only a small fraction (around 300) initially labeled. The tweets are primarily in English, though some contain local languages and emojis, which are handled during preprocessing.

## Key Features

* **Data Preprocessing:** Comprehensive cleaning of tweets, including removal of emojis, special characters, lowercasing, tokenization, and lemmatization.
* **Semi-Supervised Learning:** A core methodology to address the limited labeled data. The models are initially trained on the small labeled dataset, then used to predict labels for the vast unlabeled data, and subsequently retrained on the combined dataset to improve performance.
* **Multiple Machine Learning Models:** Implementation and evaluation of various classification algorithms:
    * **Support Vector Machine (SVM):** Utilized for its effectiveness in classification tasks.
    * **Random Forest Classifier:** A robust ensemble method known for handling complex datasets.
    * **XGBoost Classifier:** A powerful gradient boosting framework.
    * **BERT (Bidirectional Encoder Representations from Transformers):** A state-of-the-art transformer-based model for natural language understanding, specifically highlighted in this project.
* **In-depth BERT Analysis:** Particular attention is given to the application and challenges of using a pre-trained BERT model on this specific dataset, considering its potential unfamiliarity with localized vocabulary.


## Methodology

The project follows a detailed methodology to tackle the sentiment analysis problem with a limited labeled dataset:

1. **Data Loading and Initial Exploration:** Tweets are loaded from an Excel file (`oxygen_related_COVID_tweets.xlsx`). Initial analysis includes examining tweet lengths and identifying major languages.
2. **Text Preprocessing:**
    * **Cleaning:** Emojis, special characters, and links are removed, and text is converted to lowercase.
    * **Tokenization:** Tweets are tokenized. For traditional models (SVM, Random Forest, XGBoost), custom WordVec embeddings are generated from the entire 70K tweet corpus. For BERT, pre-trained tokenizers are used .
    * **Lemmatization:** Words are lemmatized to their base forms.
3. **Semi-Supervised Training:**
    * The labeled data (75% for training, 25% for validation) is used for initial model training.
    * The trained model then predicts labels for the large unlabeled dataset.
    * The model is subsequently retrained on the combined labeled and (pseudo-labeled) unlabeled data. This iterative process aims to leverage the extensive unlabeled data.
4. **Model Evaluation:** Performance metrics (validation accuracy and accuracy on labeled data) are recorded for each model.

### Emphasis on BERT

BERT, being a computationally intensive model, was trained differently from the other models due to resource constraints. It was trained only once on the labeled data and not on the entire combined dataset. This limitation impacted its performance, which was lower compared to other models. The report suggests that this could be due to:

* **Single-pass Training:** Unlike other models that underwent multiple retraining iterations with pseudo-labeled data, BERT was limited to a single training pass on only the labeled data.
* **Domain Mismatch:** The pre-trained BERT model might not be well-suited for Indian vocabulary, political acronyms, and specific context found in the tweets, which could explain its lower accuracy compared to models trained or fine-tuned on the specific dataset.


## Setup Instructions

To set up and run this project, follow these steps:

1. **Clone the repository:**

```bash
git clone <project-url>
cd your-repo-name
```

*(Note: Replace `<project-url>` with the actual GitHub repository path.)*

2. **Install Dependencies:**
The project relies on standard Python data science and NLP libraries. Install them using pip:

```bash
pip install pandas numpy matplotlib seaborn regex emoji nltk scikit-learn xgboost transformers torch
```

*(Note: `torch` or `tensorflow` are required for BERT, depending on the specific implementation. `nltk` for tokenization/lemmatization, `scikit-learn` for SVM/Random Forest, `xgboost` for XGBoost, and `transformers` for BERT.)*
3. **Download NLTK Data:**
Some NLTK functionalities (like stopwords or wordnet for lemmatization) might require downloading additional data. Run the following in a Python environment:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

4. **Data Files:** Ensure the `oxygen_related_COVID_tweets.xlsx` file is present in the project's root directory or the specified data path within the notebooks.

## How to Use

The project's functionality is primarily demonstrated through the provided Jupyter Notebooks: `Data_Analysis-report.ipynb` and `Twitter-Sentiment-Analysis-code.ipynb`.

1. **Open Jupyter Notebook:**

```bash
jupyter notebook
```

2. **Navigate and Open Notebooks:** Open `Data_Analysis-report.ipynb` to view the data analysis, and `Twitter-Sentiment-Analysis-code.ipynb` to see the complete code for preprocessing, model training, and evaluation.
3. **Run Cells:** Execute the cells sequentially in the notebooks.
    * The notebooks first handle library imports and basic settings.
    * Data loading and initial preprocessing steps are then performed.
    * Subsequent sections detail feature engineering (e.g., creating WordVec embeddings or processing for BERT).
    * Model training for SVM, Random Forest, XGBoost, and BERT is carried out.
    * Finally, model evaluation and results are displayed.
4. **Experimentation:** Users can modify parameters, experiment with different preprocessing techniques, or integrate additional labeled data to potentially improve model performance, especially for BERT.

## Results

The project successfully applied semi-supervised learning to classify tweets. The Random Forest Classifier achieved the highest validation score among the models, indicating its effectiveness on this dataset (0.7457). SVM and XGBoost also performed well. BERT's performance was lower (0.5443 validation accuracy), likely due to the limited training data for this complex model and its general pre-training on different contexts.

## Future Enhancements

* **Increased Labeled Data:** Obtaining more human-labeled data would significantly improve all models, especially in combating overfitting for models like Random Forest.
* **Emoji Integration:** Developing methods to extract sentiment information from emojis could provide additional valuable features.
* **Domain-Specific BERT Training:** Training a custom BERT tokenizer or fine-tuning a BERT model on the entire 70K tweet dataset would help it better understand the specific vocabulary and context of the tweets, leading to potentially higher accuracy.
* **Iterative Semi-Supervised Training:** Implementing multiple iterations of the semi-supervised learning loop (training on combined labeled and pseudo-labeled data) for all models, including BERT, could boost performance.
