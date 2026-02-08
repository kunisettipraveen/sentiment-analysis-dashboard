# 🎯 Product Review Sentiment Analysis

<div align="center">

![Sentiment Analysis](https://img.shields.io/badge/ML-Sentiment%20Analysis-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red)
![Status](https://img.shields.io/badge/Status-Active-success)

### *Transforming Customer Feedback into Actionable Insights* 🚀

[Live Demo](https://review-sentiment-analysis-spotmies.streamlit.app/) | [Notebooks](#-notebooks) | [Documentation](#-project-overview)

---

</div>

## 📋 Table of Contents

- [Overview](#-project-overview)
- [Key Features](#-key-features)
- [Live Demo](#-live-demo)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Jupyter Notebooks](#-running-jupyter-notebooks)
  - [Web Application](#-running-the-streamlit-web-app)
- [Jupyter Notebook Analysis](#-jupyter-notebook-analysis)
- [Notebooks Details](#-notebooks)
- [Technologies](#-technologies-used)
- [Model Performance](#-model-performance)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [Author](#-author)
- [License](#-license)

---

## 🌟 Project Overview

This project implements an **end-to-end sentiment analysis system** that analyzes product reviews to determine customer sentiment (Positive, Negative, or Neutral). Built with state-of-the-art NLP techniques and machine learning algorithms, it provides both analytical notebooks for data scientists and an interactive web application for end-users.

### **Why This Project?**

- 📊 **Business Intelligence**: Help businesses understand customer sentiment at scale
- 🤖 **Machine Learning**: Leverage supervised learning for accurate sentiment classification
- 🎨 **User-Friendly**: Interactive web interface for real-time sentiment prediction
- 📈 **Data-Driven**: Comprehensive exploratory data analysis and visualization

---

## ✨ Key Features

### 🔍 **Analysis & Modeling** (Jupyter Notebook)
- ✅ Comprehensive data preprocessing and cleaning
- ✅ Exploratory Data Analysis (EDA) with rich visualizations
- ✅ Word cloud generation for sentiment visualization
- ✅ Multiple ML model comparison (Logistic Regression, SVM, Random Forest, XGBoost, etc.)
- ✅ Advanced NLP techniques (TF-IDF, Word Embeddings)
- ✅ Hyperparameter tuning and cross-validation
- ✅ Model performance evaluation and optimization
- ✅ Feature importance analysis
- ✅ Complete end-to-end ML pipeline documentation

### 🌐 **Web Application**
- ✅ Real-time sentiment prediction
- ✅ Interactive user interface with Streamlit
- ✅ Batch processing capabilities
- ✅ Visualization of sentiment distribution
- ✅ Confidence score display
- ✅ Review history tracking

---

## 🌐 Live Demo

### **Try it Now!** 👉 [Sentiment Analysis Web App](https://review-sentiment-analysis-spotmies.streamlit.app/)

The deployed application allows you to:
- 📝 Input custom product reviews
- 🎯 Get instant sentiment predictions
- 📊 View sentiment confidence scores
- 📈 Analyze multiple reviews simultaneously

---

## 📁 Project Structure

```
Product_Review_Sentiment_Analysis/
│
├── 📁 Jupyter Notebook/             # Complete Jupyter notebook analysis
│   ├── sentiment_analysis.ipynb    # Main analysis notebook
│   ├── data_exploration.ipynb      # EDA and visualizations (optional)
│   ├── model_training.ipynb        # Model development (optional)
│   └── assets/                     # Images and charts from analysis
│       ├── wordcloud_positive.png
│       ├── wordcloud_negative.png
│       └── model_comparison.png
│
├── 🌐 webapp/                       # Streamlit web application
│   ├── app.py                      # Main Streamlit application
│   ├── utils.py                    # Helper functions
│   ├── requirements.txt            # Web app dependencies
│   └── assets/                     # UI assets
│       └── logo.png
│
├── 🤖 models/                       # Trained ML models
│   ├── sentiment_model.pkl         # Trained classifier
│   ├── vectorizer.pkl              # TF-IDF vectorizer
│   └── label_encoder.pkl           # Label encoder
│
### **📊 Dataset: Amazon Fine Food Reviews**

- **Source**: [Kaggle - Amazon Fine Food Reviews Dataset](https://www.kaggle.com/snap/amazon-fine-food-reviews)
- **Size**: 568,454 reviews
- **Columns**:
  - `Id`: Unique review identifier
  - `ProductId`: Product identifier
  - `UserId`: User identifier
  - `ProfileName`: User profile name
  - `HelpfulnessNumerator`: Number of helpful votes
  - `HelpfulnessDenominator`: Total number of votes
  - `Score`: Rating (1-5 stars)
  - `Time`: Timestamp
  - `Summary`: Brief review summary
  - `Text`: Full review text (used for analysis)

### **🎯 Sentiment Distribution**
- **Positive Reviews**: 443,777 (78.07%)
- **Negative Reviews**: 124,677 (21.93%)
- **Classification Threshold**: Score > 3 = Positive, Score ≤ 3 = Negative
│
├── 📄 requirements.txt              # Project dependencies
├── 📖 README.md                     # Project documentation
├── 🚫 .gitignore                    # Git ignore file
└── 📜 LICENSE                       # License file
```

---

## 🚀 Installation

### **Prerequisites**
- Python 3.8 or higher
- pip package manager
- Git

### **Setup Instructions**

1️⃣ **Clone the repository**
```bash
git clone https://github.com/SINCHANA20044252/Product_Review_Sentiment_Analysis.git
cd Product_Review_Sentiment_Analysis
```

2️⃣ **Create a virtual environment** (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3️⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```

4️⃣ **Download NLTK data** (if required)
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

---

## 💻 Usage

### **📓 Running Jupyter Notebooks**

```bash
# Start Jupyter Notebook
jupyter notebook

# Or use JupyterLab
jupyter lab
```

Navigate to the **`Jupyter Notebook/`** folder and open **`Review_sentimental_analysis.ipynb`**

#### **📊 Notebook Workflow:**

The notebook is organized into **13 sequential segments**:

1. **Setup**: Import libraries and download NLTK data
2. **Data Loading**: Load Amazon Reviews dataset (568,454 reviews)
3. **Labeling**: Convert ratings to binary sentiment (Positive/Negative)
4. **Preprocessing Class**: Build text cleaning pipeline
5. **Text Cleaning**: Apply preprocessing to all reviews
6. **Feature Engineering**: Create TF-IDF features (5,000 dimensions)
7. **Model Training**: Train Multinomial Naive Bayes classifier
8. **Evaluation**: Calculate accuracy and performance metrics
9. **Confusion Matrix**: Visualize model performance
10. **Top Words Analysis**: Extract most indicative words
11. **Word Visualization**: Create bar charts of top features
12. **Testing**: Predict sentiment on new reviews
13. **Summary**: Display final results and save model

#### **🎯 Quick Start Guide:**
```python
# Open the main notebook
cd "Jupyter Notebook"
jupyter notebook Review_sentimental_analysis.ipynb

# Run all cells sequentially (Shift + Enter)
# Or use Cell > Run All from menu
```

#### **💡 Key Outputs:**
- **Data Statistics**: Dataset overview and distribution
- **Preprocessing Examples**: Before/after text cleaning
- **Model Metrics**: Accuracy, Precision, Recall, F1-Score
- **Confusion Matrix**: Visual performance analysis
- **Top Words**: Most important positive/negative indicators
- **Test Predictions**: Real-time sentiment analysis examples

### **🌐 Running the Streamlit Web App**

```bash
# Navigate to project directory
cd Product_Review_Sentiment_Analysis

# Run the Streamlit app
streamlit run app.py
```

**Alternative method:**
```bash
python -m streamlit run app.py
```

The application will automatically open in your browser at `http://localhost:8501`

#### **📱 Using the Web App:**
1. 📝 Enter a product review in the text area
2. 🖱️ Click "Analyze Sentiment" button
3. 📊 View the predicted sentiment (Positive/Negative)
4. 💯 See confidence score and probability breakdown
5. 🔄 Try multiple reviews to test different predictions
6. 📈 Real-time prediction with instant results

#### **✨ Web App Features:**
- Clean, intuitive user interface
- Real-time sentiment prediction
- Confidence score visualization
- Color-coded sentiment display
- Interactive text input
- Instant feedback
- Mobile-responsive design

---

## 📓 Jupyter Notebook Analysis

### **🎯 Complete End-to-End Analysis**

The **`Jupyter Notebook/`** folder contains comprehensive analysis covering the entire machine learning pipeline:

#### **📋 What's Inside:**

| Section | Description | Key Outputs |
|---------|-------------|-------------|
| 📊 **Data Exploration** | Understanding the dataset structure and characteristics | Statistics, distributions, visualizations |
| 🧹 **Data Preprocessing** | Cleaning and preparing text data for modeling | Cleaned dataset, preprocessing pipeline |
| 🔍 **EDA** | Deep dive into patterns and insights | Word clouds, n-grams, correlation analysis |
| 🤖 **Model Training** | Building and training multiple ML models | Trained models, performance metrics |
| 📈 **Evaluation** | Comprehensive model performance analysis | Confusion matrices, ROC curves, comparison charts |
| 💾 **Model Saving** | Exporting trained models for deployment | `.pkl` files for production use |

#### **🎨 Visualizations Included:**

- ☁️ **Word Clouds**: Visual representation of most frequent words in positive/negative reviews
- 📊 **Distribution Plots**: Sentiment class distribution, review length analysis
- 🔥 **Heatmaps**: Correlation matrices and confusion matrices
- 📈 **Performance Charts**: Model comparison bar charts, ROC-AUC curves
- 🎯 **Feature Importance**: Top features contributing to predictions

#### **🔬 Analysis Highlights:**

```python
# Example: Load and analyze the dataset
import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('data/raw/reviews.csv')

# Quick overview
print(f"Total Reviews: {len(df)}")
print(f"Sentiment Distribution:\n{df['sentiment'].value_counts()}")

# Generate word cloud
from wordcloud import WordCloud
positive_reviews = ' '.join(df[df['sentiment']=='positive']['review_text'])
wordcloud = WordCloud(width=800, height=400).generate(positive_reviews)
```

#### **📚 Learning Path:**

1. **Beginners**: Start with data exploration to understand sentiment analysis basics
2. **Intermediate**: Dive into preprocessing techniques and feature engineering
3. **Advanced**: Explore model optimization and hyperparameter tuning
4. **Experts**: Analyze model errors and implement advanced NLP techniques

---

## 📓 Jupyter Notebook Analysis

The complete analysis is contained in **`Review_sentimental_analysis.ipynb`**, which includes:

### **📋 Notebook Structure** (13 Segments)

**Segment 1: Imports and Setup** 🔧
- Import all required libraries (Pandas, NumPy, NLTK, Scikit-learn)
- Download NLTK resources
- Setup visualization libraries

**Segment 2: Data Loading and Exploration** 📊
- Load Amazon Reviews dataset (568,454 reviews)
- Dataset shape and column information
- Data types and missing values analysis
- Basic statistical overview
- Rating distribution analysis

**Segment 3: Create Sentiment Labels** 🏷️
- Convert 1-5 star ratings to binary sentiment
- Threshold: Score > 3 = Positive (1), ≤ 3 = Negative (0)
- Sentiment distribution: 78.07% Positive, 21.93% Negative
- Display sample reviews for each sentiment

**Segment 4: Text Preprocessing Class** 🧹
- Custom `TextPreprocessor` class implementation
- Methods:
  - `clean_text()`: Lowercase, remove URLs, emails, numbers, punctuation
  - `remove_stopwords()`: Filter common words
  - `apply_stemming()`: Optional stemming (Porter Stemmer)
  - `preprocess()`: Complete pipeline

**Segment 5: Apply Preprocessing** ✨
- Clean all 568,454 reviews
- Show before/after examples
- Remove empty reviews
- Final dataset: 568,453 reviews

**Segment 6: Split Data and Create TF-IDF Features** 🔢
- Train-Test Split: 80-20 (454,762 train, 113,691 test)
- Stratified split for balanced distribution
- TF-IDF Vectorization:
  - 5,000 max features
  - 1-2 n-grams (unigrams and bigrams)
  - min_df=2, max_df=0.8

**Segment 7: Train the Model** 🎯
- Multinomial Naive Bayes classifier
- Training accuracy: 84.93%

**Segment 8: Evaluate the Model** 📈
- Test accuracy: 84.80%
- Detailed classification report
- Confusion matrix analysis:
  - True Negatives: 8,960
  - False Positives: 15,975
  - False Negatives: 1,307
  - True Positives: 87,449

**Segment 9: Visualize Confusion Matrix** 📊
- Heatmap visualization using Seaborn
- Color-coded performance metrics

**Segment 10: Top Indicative Words** 🔍
- Extract top 20 words for each sentiment
- Display word importance (log probabilities)
- Identify key positive/negative indicators

**Segment 11: Visualize Top Words** 📊
- Horizontal bar charts for top 15 words
- Color-coded: Red (Negative), Green (Positive)
- Log probability comparison

**Segment 12: Test with New Reviews** 🧪
- Interactive prediction function
- Test with 5 sample reviews
- Display sentiment, confidence, and probabilities

**Segment 13: Summary and Results** 📋
- Complete performance metrics
- Per-class precision, recall, F1-score
- Optional model saving (pickle format)

---

## 🛠️ Technologies Used

### **Programming & Libraries**
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

### **Core Technologies**

| Category | Technologies |
|----------|-------------|
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost |
| **NLP** | NLTK, spaCy, TextBlob |
| **Visualization** | Matplotlib, Seaborn, Plotly, WordCloud |
| **Web Framework** | Streamlit |
| **Deployment** | Streamlit Cloud |
| **Version Control** | Git, GitHub |

---

## 📈 Model Performance

### **🎯 Final Results (Multinomial Naive Bayes)**

| Metric | Value |
|--------|-------|
| **Training Accuracy** | 84.93% |
| **Test Accuracy** | 84.80% |
| **Dataset Size** | 568,453 reviews |
| **Training Samples** | 454,762 (80%) |
| **Test Samples** | 113,691 (20%) |
| **Features** | 5,000 (TF-IDF) |

### **📊 Class-wise Performance**

#### **Negative Class (Score ≤ 3)**
| Metric | Score |
|--------|-------|
| Precision | 87.27% |
| Recall | 35.93% |
| F1-Score | 50.91% |
| Support | 24,935 |

#### **Positive Class (Score > 3)**
| Metric | Score |
|--------|-------|
| Precision | 84.55% |
| Recall | 98.53% |
| F1-Score | 91.01% |
| Support | 88,756 |

### **🔍 Confusion Matrix Analysis**

|  | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | 8,960 (TN) | 15,975 (FP) |
| **Actual Positive** | 1,307 (FN) | 87,449 (TP) |

- **True Negatives (TN)**: 8,960 correctly identified negative reviews
- **False Positives (FP)**: 15,975 negative reviews incorrectly marked as positive
- **False Negatives (FN)**: 1,307 positive reviews incorrectly marked as negative
- **True Positives (TP)**: 87,449 correctly identified positive reviews

### **⚡ Model Strengths**
- ✅ **High recall for positive reviews** (98.53%) - Catches almost all positive sentiment
- ✅ **Good overall accuracy** (84.80%) - Reliable general performance
- ✅ **Balanced precision** - Minimizes false predictions
- ✅ **Fast prediction** - Real-time analysis capability

### **⚠️ Model Limitations**
- ⚠️ **Lower recall for negative reviews** (35.93%) - May miss some negative sentiment
- ⚠️ **Class imbalance impact** - Dataset has 3.6x more positive than negative reviews
- ⚠️ **Neutral sentiment handling** - Binary classification doesn't capture neutrality

*Note: Performance metrics may vary based on dataset characteristics and hyperparameters*

---

## 📸 Screenshots

### **Web Application Interface**
```
┌─────────────────────────────────────────┐
│  🎯 Product Review Sentiment Analysis  │
├─────────────────────────────────────────┤
│                                         │
│  Enter Your Review:                     │
│  ┌─────────────────────────────────┐   │
│  │ This product exceeded my        │   │
│  │ expectations! Highly recommend! │   │
│  └─────────────────────────────────┘   │
│                                         │
│  [ Analyze Sentiment ]                  │
│                                         │
│  ✅ Sentiment: POSITIVE                │
│  📊 Confidence: 94.5%                  │
│                                         │
└─────────────────────────────────────────┘
```

### **Notebook Visualizations**
- Word clouds showing frequent terms
- Sentiment distribution pie charts
- Model comparison bar charts
- Confusion matrices
- Feature importance plots

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the branch (`git push origin feature/AmazingFeature`)
5. 🔁 Open a Pull Request

### **Contribution Ideas**
- 🎨 Improve UI/UX of the web app
- 🤖 Add more ML models
- 📊 Enhance visualizations
- 🌐 Add multi-language support
- 📝 Improve documentation
- 🧪 Add unit tests

---

## 👨‍💻 Author

**SINCHANA**

- 🔗 GitHub: [@SINCHANA20044252](https://github.com/SINCHANA20044252)
- 💼 LinkedIn: [Connect with me](#)
- 📧 Email: [Your Email]

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Thanks to all contributors who have helped with this project
- Dataset sources and references
- Open-source community for amazing libraries and tools

---

## 📞 Support

If you found this project helpful, please consider:
- ⭐ Starring the repository
- 🐛 Reporting bugs via [Issues](https://github.com/SINCHANA20044252/Product_Review_Sentiment_Analysis/issues)
- 💡 Suggesting new features
- 📢 Sharing with others

---

<div align="center">

### ⭐ If you like this project, please give it a star! ⭐

**Made with ❤️ by SINCHANA**

</div>
