import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import precision_score , accuracy_score
from scipy.stats import ttest_ind
import time
# Load data
csv_data_path = r'C:\Users\linus\Desktop\Python\bookfodder\BX-Book-Ratings.csv'
csv_file_path = r'C:\Users\linus\Desktop\Python\bookfodder\BX-Users.csv'


column_names1 = ['user_id','ISBN','Book-Rating']
ratings = pd.read_csv(csv_data_path,sep='";',header=None, names=column_names1, engine='python', encoding='latin-1')
ratings.head()


user_cols = ['user_id', 'Location', 'Age',]
users = pd.read_csv(csv_file_path, sep='";', names=user_cols, skiprows=[0], engine='python', encoding='latin-1')
users = users[['user_id', 'Age']]
users.head()


data = pd.merge(ratings, users, on='user_id')
# Replace 'NULL' with NaN
# Replace 'NULL' with NaN
# Remove quotes from string values in 'Age' column
data['Age'] = data['Age'].str.strip('"')


# Convert 'Age' column to numeric
data['Age'] = pd.to_numeric(data['Age'], errors='coerce')


data['Age'].replace('NULL', np.nan, inplace=True)


# Drop rows with NaN values in the 'Age' column
data.dropna(subset=['Age'], inplace=True)
data['Age'] = pd.to_numeric(data['Age'])




# Compute the most popular movies overall
book_popularity = defaultdict(int)
for book in data['ISBN']:
    book_popularity[book] += 1
most_popular = sorted(book_popularity.items(), key=lambda x: x[1], reverse=True)


# Split the data into male and female subsets
data_Age_below = data[data['Age'] <= 30]
data_Age_above = data[data['Age'] >= 31]


# Compute the most popular movies for male and female subsets
book_popularity_below = defaultdict(int)
for book in data_Age_below['ISBN']:
    book_popularity_below[book] += 1
most_popular_below = sorted(book_popularity_below.items(), key=lambda x: x[1], reverse=True)


book_popularity_above = defaultdict(int)
for book in data_Age_above['ISBN']:
    book_popularity_above[book] += 1
most_popular_above = sorted(book_popularity_above.items(), key=lambda x: x[1], reverse=True)
# Compute precision for male and female subsets using the most popular movies
k = 10
below_users = data_Age_below['user_id'].unique()
above_users = data_Age_above['user_id'].unique()
precision_below = []
precision_above = []
print(len(below_users))
print(len(above_users))


for user in below_users:
    print(f"Processing user {user}...")
    books_watched = data_Age_below[data_Age_below['user_id'] == user]['ISBN']
    recommendations = [book[0] for book in most_popular_below][:k]
    relevant_books = list(set(books_watched) & set(recommendations))
    if len(recommendations) > 0:
        precision_below.append(len(relevant_books) / len(recommendations))


for user in above_users:
    print(f"Processing user {user}...")
    books_watched = data_Age_above[data_Age_above['user_id'] == user]['ISBN']
    recommendations = [book[0] for book in most_popular_above][:k]
    relevant_books = list(set(books_watched) & set(recommendations))
    if len(recommendations) > 0:
        precision_above.append(len(relevant_books) / len(recommendations))


# Compute t-test to compare precision between male and female users
t_statistic, p_value = ttest_ind(precision_below, precision_above)


# Print results
print("Precision (below):", np.mean(precision_below))
print("Precision (above):", np.mean(precision_above))
print("T-test statistic:", t_statistic)
print("P-value:", p_value)


def check_demographic_parity(data, attribute, k=10):
    attribute_values = data[attribute].unique()
    precision_values = []


    for value in attribute_values:
        data_subset = data[data[attribute] == value]
        book_popularity_subset = defaultdict(int)
        for book in data_subset['ISBN']:
            book_popularity_subset[book] += 1
        most_popular_subset = sorted(book_popularity_subset.items(), key=lambda x: x[1], reverse=True)
        users_subset = data_subset['user_id'].unique()
        precision_subset = []
   
        for user in users_subset:
            books_watched = data_subset[data_subset['user_id'] == user]['ISBN']
            recommendations = [book[0] for book in most_popular_subset][:k]
            relevant_books = list(set(books_watched) & set(recommendations))
            if len(recommendations) > 0:
                precision_subset.append(len(relevant_books) / len(recommendations))
   
        mean_precision_subset = np.mean(precision_subset)
        precision_values.append(mean_precision_subset)


    return precision_values


precision_AGE = check_demographic_parity(data, 'Age')
gender_parity = np.abs(precision_AGE[0] - precision_AGE[1])


if gender_parity < 0.1:
    print("There is no evidence of gender bias in the recommendation system.")
else:
    print("There may be gender bias in the recommendation system.")


def compute_accuracy(data_subset, most_popular_subset, k):
    users_subset = data_subset['user_id'].unique()
    accuracy = []


    for user in users_subset:
        books_watched = data_subset[data_subset['user_id'] == user]['ISBN']
        recommendations = [book[0] for book in most_popular_subset][:k]
        relevant_books = list(set(books_watched) & set(recommendations))
        accuracy.append(len(relevant_books) / len(books_watched))


    return np.mean(accuracy)


k = 15


accuracy_below = compute_accuracy(data_Age_below, most_popular_below, k)
accuracy_above = compute_accuracy(data_Age_above, most_popular_above, k)
accuracy_parity = accuracy_above - accuracy_below


print("Accuracy (male):", accuracy_below)
print("Accuracy (female):", accuracy_above)
print("Accuracy Parity", accuracy_parity)
