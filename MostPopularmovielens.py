import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import precision_score , accuracy_score
from scipy.stats import ttest_ind

# Load data
csv_data_path = r'C:\Users\linus\Desktop\Python\ml-1m\ratings.dat'
csv_file_path = r'C:\Users\linus\Desktop\Python\ml-1m\users.dat'

column_names1 = ['user_id','movie_id','rating','timestamp']
ratings = pd.read_csv(csv_data_path, sep='::', header=None, names=column_names1, engine='python')

user_cols = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
users = pd.read_csv(csv_file_path, sep='::', names=user_cols, engine='python')
users = users[['user_id', 'gender']]


data = pd.merge(ratings, users, on='user_id')


# Compute the most popular movies overall
movie_popularity = defaultdict(int)
for movie in data['movie_id']:
    movie_popularity[movie] += 1
most_popular = sorted(movie_popularity.items(), key=lambda x: x[1], reverse=True)


# Split the data into male and female subsets
data_male = data[data['gender'] == 'M']
data_female = data[data['gender'] == 'F']


# Compute the most popular movies for male and female subsets
movie_popularity_male = defaultdict(int)
for movie in data_male['movie_id']:
    movie_popularity_male[movie] += 1
most_popular_male = sorted(movie_popularity_male.items(), key=lambda x: x[1], reverse=True)


movie_popularity_female = defaultdict(int)
for movie in data_female['movie_id']:
    movie_popularity_female[movie] += 1
most_popular_female = sorted(movie_popularity_female.items(), key=lambda x: x[1], reverse=True)


# Compute precision for male and female subsets using the most popular movies
k = 10
male_users = data_male['user_id'].unique()
female_users = data_female['user_id'].unique()
precision_male = []
precision_female = []
print(len(male_users))
print(len(female_users))
for user in male_users:
    movies_watched = data_male[data_male['user_id'] == user]['movie_id']
    recommendations = [movie[0] for movie in most_popular_male][:k]
    relevant_movies = list(set(movies_watched) & set(recommendations))
    if len(recommendations) > 0:
        precision_male.append(len(relevant_movies) / len(recommendations))


for user in female_users:
    movies_watched = data_female[data_female['user_id'] == user]['movie_id']
    recommendations = [movie[0] for movie in most_popular_female][:k]
    relevant_movies = list(set(movies_watched) & set(recommendations))
    if len(recommendations) > 0:
        precision_female.append(len(relevant_movies) / len(recommendations))


# Compute t-test to compare precision between male and female users
t_statistic, p_value = ttest_ind(precision_male, precision_female)


# Print results
print("Precision (male):", np.mean(precision_male))
print("Precision (female):", np.mean(precision_female))
print("T-test statistic:", t_statistic)
print("P-value:", p_value)


def check_demographic_parity(data, attribute, k=10):
    attribute_values = data[attribute].unique()
    precision_values = []


    for value in attribute_values:
        data_subset = data[data[attribute] == value]
        movie_popularity_subset = defaultdict(int)
        for movie in data_subset['movie_id']:
            movie_popularity_subset[movie] += 1
        most_popular_subset = sorted(movie_popularity_subset.items(), key=lambda x: x[1], reverse=True)
        users_subset = data_subset['user_id'].unique()
        precision_subset = []
   
        for user in users_subset:
            movies_watched = data_subset[data_subset['user_id'] == user]['movie_id']
            recommendations = [movie[0] for movie in most_popular_subset][:k]
            relevant_movies = list(set(movies_watched) & set(recommendations))
            if len(recommendations) > 0:
                precision_subset.append(len(relevant_movies) / len(recommendations))
   
        mean_precision_subset = np.mean(precision_subset)
        precision_values.append(mean_precision_subset)


    return precision_values


precision_gender = check_demographic_parity(data, 'gender')
gender_parity = np.abs(precision_gender[0] - precision_gender[1])


if gender_parity < 0.1:
    print("There is no evidence of gender bias in the recommendation system.")
else:
    print("There may be gender bias in the recommendation system.")


def compute_accuracy(data_subset, most_popular_subset, k):
    users_subset = data_subset['user_id'].unique()
    accuracy = []


    for user in users_subset:
        movies_watched = data_subset[data_subset['user_id'] == user]['movie_id']
        recommendations = [movie[0] for movie in most_popular_subset][:k]
        relevant_movies = list(set(movies_watched) & set(recommendations))
        accuracy.append(len(relevant_movies) / len(movies_watched))


    return np.mean(accuracy)

k = 15

accuracy_male = compute_accuracy(data_male, most_popular_male, k)
accuracy_female = compute_accuracy(data_female, most_popular_female, k)
accuracy_parity = accuracy_female - accuracy_male

print("Accuracy (male):", accuracy_male)
print("Accuracy (female):", accuracy_female)
print("Accuracy Parity", accuracy_parity)
