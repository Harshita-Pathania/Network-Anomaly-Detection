import pandas as pd
from pymongo import MongoClient
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Connect to MongoDB
client = MongoClient(
    '',
    connectTimeoutMS=30000,
    socketTimeoutMS=30000
)
db = client['NB15']
train_collection = db['UNSW_Tr']
test_collection = db['UNSW_Te']

# Fetch a larger sample from MongoDB to ensure diversity
sample_size = 10000
train_data = pd.DataFrame(list(train_collection.aggregate([{ '$sample': { 'size': sample_size } }])))
test_data = pd.DataFrame(list(test_collection.aggregate([{ '$sample': { 'size': sample_size // 2 } }])))

# Drop the MongoDB ID fields
train_data.drop(columns=['_id'], inplace=True)
test_data.drop(columns=['_id'], inplace=True)

# Combine train and test data for stratified sampling
full_data = pd.concat([train_data, test_data], ignore_index=True)

# Ensure at least 2 instances of each attack type in both train and test sets
min_class_size = 2
def ensure_min_class_size(df, label_col, min_size):
    # Count instances per class
    counts = df[label_col].value_counts()
    # Classes that need more samples
    needs_more_samples = counts[counts < min_size].index
    for cls in needs_more_samples:
        # Add more samples for each class that is underrepresented
        additional_samples = df[df[label_col] == cls].sample(n=min_size - counts[cls], replace=True)
        df = pd.concat([df, additional_samples], ignore_index=True)
    return df

# Ensure both train and test sets have at least min_class_size instances of each attack type
train_sample = ensure_min_class_size(train_data, 'attack_cat', min_class_size)
test_sample = ensure_min_class_size(test_data, 'attack_cat', min_class_size)

# Perform stratified sampling
train_sample, test_sample = train_test_split(
    full_data, test_size=0.2, stratify=full_data['attack_cat'], random_state=42
)

# Relevant features for anomaly detection
relevant_features = [
    'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 
    'sload', 'dload', 'sloss', 'dloss', 'sjit', 'djit', 'ct_srv_src', 
    'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 
    'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 
    'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports'
]

# Encode categorical features
categorical_features = ['proto', 'service', 'state']
train_sample = pd.get_dummies(train_sample, columns=categorical_features)
test_sample = pd.get_dummies(test_sample, columns=categorical_features)

# Ensure both train and test have the same columns after encoding
missing_cols = set(train_sample.columns) - set(test_sample.columns)
for col in missing_cols:
    test_sample[col] = 0
test_sample = test_sample[train_sample.columns]

# Update relevant_features to include all encoded categorical features
relevant_features += [col for col in train_sample.columns if col.startswith(tuple(categorical_features))]

# Separate features and labels
X_train = train_sample[relevant_features]
y_train = train_sample['attack_cat']
X_test = test_sample[relevant_features]
y_test = test_sample['attack_cat']

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)
y_prob = knn.predict_proba(X_test)

# Debug: Check the shape of y_prob
print("Shape of y_prob:", y_prob.shape)

# Display the results
print(f"Total activities: {len(y_test)}")
anomaly_info = []
for i, (index, pred, prob) in enumerate(zip(np.arange(len(y_test)), y_pred, y_prob)):
    actual_attack = y_test.iloc[index]
    record_id = test_sample.iloc[index]['id']  # Assuming 'id' column exists in the dataset
    prob_dict = {knn.classes_[i]: p for i, p in enumerate(prob)}
    predicted_prob = prob_dict[pred]
    anomaly_info.append((index, record_id, actual_attack, pred, predicted_prob))
    print(f"Activity {i+1}: ID: {record_id}, Actual Attack: {actual_attack}, Predicted: {pred}, Probability: {predicted_prob}")

# Plot the probability of each network activity being an anomaly
def plot_probabilities(anomaly_info):
    plt.figure(figsize=(14, 8))
    
    # Determine the level of probability
    def determine_level(prob):
        if prob < 0.33:
            return 'Low'
        elif prob < 0.66:
            return 'Medium'
        else:
            return 'High'
    
    # Colors for each probability level
    level_colors = {'Low': 'blue', 'Medium': 'orange', 'High': 'red'}
    attack_colors = sns.color_palette('hsv', len(set(y_test)))  # Unique color for each attack type
    
    # Attack type to color mapping
    attack_color_map = {attack: attack_colors[i] for i, attack in enumerate(set(y_test))}
    
    plotted_levels = {level: False for level in level_colors}
    plotted_attacks = {attack: False for attack in attack_color_map}
    
    for index, record_id, actual_attack, pred, predicted_prob in anomaly_info:
        level = determine_level(predicted_prob)
        color = level_colors[level]
        attack_color = attack_color_map[actual_attack]
        plt.scatter(index, predicted_prob, color=color, edgecolor=attack_color, label=f'{actual_attack}: ID {record_id}' if not plotted_attacks[actual_attack] else "")
        plotted_levels[level] = True
        plotted_attacks[actual_attack] = True
    
    # Add legend for probability levels
    level_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=level) for level, color in level_colors.items()]
    attack_handles = [plt.Line2D([0], [0], marker='o', color='w', markeredgecolor=color, markersize=10, label=attack) for attack, color in attack_color_map.items()]
    
    plt.legend(handles=level_handles + attack_handles, title='Legend', loc='upper right')
    plt.xlabel('Network Activity Index')
    plt.ylabel('Probability of being an attack')
    plt.title('Probability of each Network Activity being an Anomaly')

    plt.show()

# Plot for all attack types
plot_probabilities(anomaly_info)
