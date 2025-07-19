Company : CODTECH IT solutions PVT LTD
Name : Aditya bisen 
Intern id : CT06DF1711 
Domain : Machine learning 
Duration : 6 weeks 
Mentor : Neela Santhosh Kumar

# Sentiment_analysis_logistic
This project focuses on performing Sentiment Analysis using machine learning techniques to classify customer reviews as either positive or negative. Sentiment analysis, a key application of Natural Language Processing (NLP), helps in understanding public opinion, customer satisfaction, and brand perception by analyzing textual feedback.

In this project, a machine learning pipeline is built using TF-IDF vectorization for feature extraction and Logistic Regression for classification. The goal is to train a model that can predict the sentiment of unseen reviews with high accuracy and interpretability.

🧠 Objective
The primary aim is to:

Convert unstructured text data (customer reviews) into meaningful numerical features using TF-IDF.

Train a logistic regression model to classify the sentiments of reviews.

Evaluate the model’s performance on real-world data using common metrics like accuracy, precision, recall, and confusion matrix.

📊 Dataset
The dataset consists of customer reviews, each labeled with a sentiment (positive or negative). Each review is a raw text entry that expresses an opinion or feedback on a product, service, or experience.

Example format:

Review	Sentiment
"The product quality is amazing"	Positive
"I’m very disappointed"	Negative

You can use any open-source sentiment dataset, such as the IMDB Movie Reviews, Amazon Product Reviews, or a custom review dataset in .csv format.

🛠 Tools & Libraries Used
Python: The programming language used to write and execute code.

Jupyter Notebook / VS Code: For interactive coding and result visualization.

Pandas: To handle data loading, cleaning, and manipulation.

NumPy: For numerical operations.

Scikit-learn:

TfidfVectorizer – For converting text into numerical TF-IDF features.

LogisticRegression – To train the sentiment classifier.

train_test_split, accuracy_score, classification_report, confusion_matrix – For model evaluation and testing.

Matplotlib & Seaborn: For visualizing confusion matrix and result metrics.

🧪 Model Building Steps
Data Preprocessing

Load the CSV file and clean the text (e.g., remove punctuation, lowercase, stopwords).

Encode the target sentiment labels to numerical values (e.g., 1 = Positive, 0 = Negative).

Feature Extraction

Apply TF-IDF Vectorization to convert text into numerical vectors that reflect the importance of words relative to the entire corpus.

Model Training

Use Logistic Regression, a fast and effective linear model, to learn patterns in the TF-IDF features that indicate sentiment.

Evaluation

Measure model performance using metrics like accuracy, precision, recall, and F1-score.

Visualize results using a confusion matrix.

✅ Output & Results
The final notebook produces:

A trained model capable of predicting sentiment from raw text input.

TF-IDF vectorized data used for training and testing.

Performance metrics and visualizations.

Insights into which words or terms contributed most to predictions.

📦 Deliverables
A Jupyter Notebook with:

Preprocessing steps

Model building and training

Evaluation and visualization

Clean code with comments for easy understanding

output

<img width="805" height="787" alt="Image" src="https://github.com/user-attachments/assets/db1c6c95-33ee-412c-813d-6882d1814bc5" />



