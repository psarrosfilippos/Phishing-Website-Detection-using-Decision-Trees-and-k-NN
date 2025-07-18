## Phishing Website Detection using Decision Trees and k-NN

This project focuses on detecting phishing websites through supervised machine learning. Two classifiers were implemented and evaluated: Decision Trees and k-Nearest Neighbors (k-NN). The dataset underwent preprocessing with normalization and was split using the train-test split method. The algorithms were optimized via hyperparameter tuning and evaluated based on accuracy, recall, and F1 score.

A detailed comparison of their performance is provided, highlighting strengths and weaknesses of each approach in detecting phishing attempts accurately.
## Features
**ARFF Dataset Loader:** Uses scipy.io.arff to load .arff format datasets.

**Data Preprocessing:**

- Label encoding for categorical class labels

- Feature scaling using StandardScaler

**Train-Test Split:** 80/20 split for training and evaluation.

**Decision Tree Classifier:**

- Hyperparameter tuning for max_leaf_nodes

- Confusion matrix and performance metrics (Accuracy, Recall, F1 Score)

**k-Nearest Neighbors (k-NN):**

- Optimization of k from 1 to 30

- Evaluation using standard classification metrics

**Performance Visualization:**

- Confusion matrices (heatmaps)

- Accuracy vs. hyperparameter plots

- Comparative bar charts and tabular summaries for both models

**Modular Codebase:** Easy to modify and experiment with other classifiers or datasets.

## Technologies Used

**Python 3.10+:** Core programming language used for the entire implementation

**NumPy:** Efficient numerical computations and array manipulation

**Pandas:** Data loading, handling, and analysis

**scikit-learn (sklearn):**

- Classification models: Decision Tree, k-Nearest Neighbors

- Preprocessing tools: LabelEncoder, StandardScaler

- Model evaluation: train-test split, accuracy, recall, F1 score

**Matplotlib & Seaborn:** Visualization of confusion matrices, performance metrics, and result tables

**SciPy:** Loading of ARFF (Attribute-Relation File Format) datasets (scipy.io.arff)
## How to Use

**1. Clone the Repository**

Download the project to your local machine:

    git clone https://github.com/your-username/phishing-detection-classifiers.git
    cd phishing-detection-classifiers

Alternatively, you can download the .zip file and extract it manually

**2. Install Dependencies**

Install the required Python packages using pip:

    pip install -r requirements.txt

Or, install individually if you don’t have a requirements.txt:

    pip install numpy pandas matplotlib seaborn scikit-learn scipy

**3. Place Your Dataset**

Make sure the dataset file (e.g. Training Dataset.arff) is placed in the same directory as the script.
If your dataset has a different name or format, update the corresponding path in the script:

    from scipy.io import arff
    data, meta = arff.loadarff('Training Dataset.arff')

**4. Run the Script**

Execute the script using:

    python phishing_detection.py

The script will:

- Load and preprocess the dataset

- Train Decision Tree and k-NN classifiers

- Evaluate performance using accuracy, recall, and F1 score

- Display confusion matrices and comparative plots

**5. Customize (Optional)**

If you want to experiment:

- Add your own dataset (in .arff format)

- Adjust hyperparameters (e.g. max_leaf_nodes, k)

- Modify visualizations or metrics in the code






## Example Output

After running the script, you will see results printed in the terminal and visualizations showing model performance. Example output for both Decision Tree and k-NN classifiers might look like:

    best Decision Tree: max_leaf_nodes=201, Accuracy=0.9583
    best k-NN: k=1, Accuracy=0.9574

The following will also be displayed:

**Confusion Matrices:**

- Visual heatmaps showing true positives, false positives, true negatives, and false negatives.

**Performance Metrics (Accuracy, Recall, F1 Score):**

- Bar charts comparing both algorithms.

**Summary Tables:**

- Showing performance for each max_leaf_nodes (Decision Tree) and k (k-NN).

**Line Graphs:**

- Accuracy vs max_leaf_nodes (for Decision Tree)

- Accuracy vs k (for k-NN)

**Sample Classification Report (printed):**

    Decision Tree Classifier
    Accuracy: 0.9583
    Recall: 0.9737
    F1 Score: 0.9637

    k-NN Classifier
    Accuracy: 0.9574
    Recall: 0.9681
    F1 Score: 0.9627

These outputs provide a visual and statistical comparison of both algorithms on the phishing detection dataset.

## Acknowledgements

This project was developed as part of my final-year coursework during the fourth year of my undergraduate studies in Informatics and Telecommunications.

I would like to thank my professor for his valuable guidance and feedback, which helped shape the direction and rigor of this work.

Special appreciation goes to the academic staff and peers who contributed through discussions and insights, as well as to the open-source community — particularly the developers of NumPy, scikit-learn, and Matplotlib — whose tools enabled the implementation of this project.



## Authors

Filippos Psarros

informatics and telecommunications Student

GitHub: psarrosfilippos

[README.md](https://github.com/user-attachments/files/21316570/README.md)
