# Automatic_Ticket_Classification_NLP
This project implements an automated ticket classification system for financial customer complaints, using unsupervised NMF for topic discovery and supervised learning for prediction, achieving 95% accuracy with logistic regression to automate routing of complaints to the appropriate departments.


## Project Overview
This project addresses the challenge of automatically classifying customer complaints in the financial industry based on their content. By implementing a two-stage machine learning approach combining unsupervised topic modeling and supervised classification, the system can accurately categorize incoming complaints into five distinct categories, enabling faster resolution by routing them to the correct departments.

## Dataset Description
The dataset consists of customer complaints submitted to financial institutions in JSON format. These unstructured text complaints cover various issues related to banking services, credit cards, loans, and other financial products. Each complaint contains the actual text of the customer issue along with metadata. Personal identifying information in the dataset has been masked as "XXXX" for privacy and was removed during preprocessing.

## Implementation Pipeline
The project follows a structured data science workflow:

1. **Data Loading**: Loading and parsing the JSON data into a pandas DataFrame
2. **Text Preprocessing**: Cleaning and normalizing text through lowercase conversion, punctuation removal, lemmatization, and POS filtering
3. **Exploratory Data Analysis**: Analyzing complaint length distributions and key n-grams
4. **Feature Extraction**: Converting text to numerical representations using TF-IDF vectorization
5. **Topic Modeling**: Using Non-negative Matrix Factorization (NMF) to discover latent topics
6. **Model Building**: Training supervised classification models on the discovered topics
7. **Model Evaluation**: Comparing model performance using accuracy, precision, recall, and F1-score
8. **Model Inference**: Testing the model on new complaint examples

## Methodology

### Text Preprocessing
- Converted text to lowercase
- Removed punctuation and special characters
- Eliminated words containing numbers
- Applied lemmatization to reduce words to their base forms
- Performed part-of-speech filtering to retain nouns (most informative for classification)
- Removed masked personal information (XXXX)

### Topic Modeling
Using Non-negative Matrix Factorization (NMF) with 5 components, we successfully identified distinct topic clusters in the complaints data. The model examined word co-occurrence patterns to group complaints into coherent categories.

### Supervised Classification
Four machine learning models were implemented and compared:
- Logistic Regression
- Random Forest
- Decision Tree
- Naive Bayes

## Results and Findings

### Topic Identification
The NMF model successfully identified 5 distinct complaint categories:
1. **Credit Card / Prepaid Card**: Issues related to credit card transactions, fees, and disputes
2. **Credit Reporting / Identity Theft**: Problems with credit reports, scores, and identity theft
3. **Mortgages/Loans**: Concerns about loan applications, terms, and modifications
4. **Bank Account Services**: Problems with checking/savings accounts and banking operations
5. **Payment Issues**: Problems with payment processing, late fees, and billing

### Model Performance

| Model | Accuracy | Notes |
|-------|----------|-------|
| Logistic Regression | 95% | Best overall performance, balanced metrics across all categories |
| Random Forest | 84% | Good performance but struggles with recall for category 4 (57%) |
| Decision Tree | 80% | Consistent performance but less accurate than top models |
| Naive Bayes | 35% | Poor performance across all metrics |

The confusion matrix analysis confirms the strong performance of the logistic regression model, with high diagonal values indicating excellent classification accuracy across all categories.

### Key Feature Insights

**Credit Card / Prepaid Card Keywords:**  
Terms like "charge", "dispute", "card", and "transaction" are strong predictors. These terms clearly relate to credit card transactions and dispute resolution.

**Credit Reporting / Identity Theft:**  
Words like "credit", "report", "inquiry", and "score" have high predictive power, indicating issues with credit reporting and monitoring.

**Mortgages/Loans Keywords:**  
Terms such as "loan", "mortgage", "home", and "modification" strongly indicate loan-related complaints, reflecting concerns about lending products and terms.

**Bank Account Services:**  
Words like "account", "check", "bank", and "deposit" are key indicators of bank account service issues, covering routine banking operations.

**Payment Issues:**  
Terms like "payment", "late", "fee", and "balance" signal payment-related concerns, often regarding fees and timing of payments.

## Conclusion

The project successfully demonstrates an effective approach to automatically classify customer complaints in the financial sector. The logistic regression model achieves 95% accuracy with excellent precision and recall across all categories, making it the optimal choice for this task.

The feature importance analysis reveals distinct vocabulary patterns for each complaint category, validating our classification approach. These insights not only improve classification accuracy but also provide valuable business intelligence about common customer issues.

This automated classification system would enable financial institutions to quickly route complaints to the appropriate departments, reducing response time and improving customer satisfaction.

## Requirements
- Python 3.6+
- Libraries: pandas, numpy, scikit-learn, nltk, spacy, matplotlib, seaborn, wordcloud
- NLTK resources: punkt, wordnet, stopwords, averaged_perceptron_tagger
- spaCy English language model: en_core_web_sm
