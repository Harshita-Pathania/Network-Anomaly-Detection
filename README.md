# KNN-based Network Anomaly Detection System

This project implements a K-Nearest Neighbors (KNN) classifier to detect anomalies in network activities using the UNSW NB15 dataset. The system fetches data from a MongoDB collection, processes it, and trains the KNN model to classify network activities into attack categories with associated probabilities.

## Project Structure

- **Data Collection**: Fetches a random sample from MongoDB using `pymongo` and processes it.
- **Data Preprocessing**: Handles categorical encoding, feature normalization, and ensures stratified sampling to avoid class imbalance.
- **Model Training**: A KNN model is trained on network activity data.
- **Anomaly Detection**: Predicts and calculates the probability of each activity being an attack.
- **Visualization**: Displays the probability of each network activity being an anomaly through a scatter plot.

## Prerequisites

1. **MongoDB Connection**: The system requires a MongoDB database connection with collections for the UNSW NB15 training and test datasets.
2. **Python Libraries**: The following libraries are required:
   - `pandas`
   - `numpy`
   - `pymongo`
   - `scikit-learn`
   - `matplotlib`
   - `seaborn`

To install the necessary dependencies, run:
```bash
pip install pandas numpy pymongo scikit-learn matplotlib seaborn
```

## How to Run

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/network-anomaly-detection.git
   cd network-anomaly-detection
   ```

2. **Update MongoDB Credentials**: Modify the connection string to your MongoDB cluster in the code.
   ```python
   client = MongoClient('your-mongo-db-connection-string')
   ```

3. **Run the Script**:
   ```bash
   python anomaly_detection.py
   ```

## Data Processing Steps

1. **Data Sampling**: 
   A sample of 10,000 training records and 5,000 test records is fetched from the MongoDB collections `UNSW_Tr` and `UNSW_Te`, respectively.
   ```python
   sample_size = 10000
   train_data = pd.DataFrame(list(train_collection.aggregate([{ '$sample': { 'size': sample_size } }])))
   test_data = pd.DataFrame(list(test_collection.aggregate([{ '$sample': { 'size': sample_size // 2 } }])))
   ```

2. **Feature Engineering**:
   - The `attack_cat` column is used as the target for classification.
   - Categorical features (`proto`, `service`, `state`) are one-hot encoded.
   - The data is split into training and testing sets using stratified sampling to ensure all attack categories are represented.
   ```python
   train_sample, test_sample = train_test_split(full_data, test_size=0.2, stratify=full_data['attack_cat'], random_state=42)
   ```

3. **Data Normalization**:
   All numerical features are normalized using `StandardScaler` for better model performance.
   ```python
   scaler = StandardScaler()
   X_train = scaler.fit_transform(X_train)
   X_test = scaler.transform(X_test)
   ```

## KNN Classifier

The KNN model is trained on the normalized training data with 5 nearest neighbors:
```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
```

## Visualization

The system provides a visual representation of the predicted probabilities of each network activity being an attack:
- Network activities are plotted against the predicted probabilities.
- Color-coding represents different levels of probability (Low, Medium, High).
- A separate color is used to highlight each attack type.

```python
def plot_probabilities(anomaly_info):
    plt.scatter(index, predicted_prob, color=color, edgecolor=attack_color, label=f'{actual_attack}: ID {record_id}')
    plt.show()
```

## Example Output

An example output of predicted probabilities for network activities is displayed in the console and visualized as a scatter plot:
```
Activity 1: ID: 12345, Actual Attack: DoS, Predicted: DoS, Probability: 0.85
Activity 2: ID: 67890, Actual Attack: Normal, Predicted: Normal, Probability: 0.15
...
```
