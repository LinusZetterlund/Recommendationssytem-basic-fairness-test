import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from scipy.stats import ttest_ind
from sklearn.metrics import precision_score, accuracy_score


csv_data_path = r'C:\Users\linus\Desktop\Python\ml-1m\ratings.dat'
csv_item_path = r'C:\Users\linus\Desktop\Python\ml-1m\movies.dat'
csv_file_path = r'C:\Users\linus\Desktop\Python\ml-1m\users.dat'


d = 'movie id | movie title | release date | video release date | Action | Adventure | Animation | Children | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western'
column_names2 = d.split(' | ')
column_names2


movies = pd.read_csv(csv_item_path, delimiter='::', encoding='latin-1', engine='python', header=None, names=column_names2)
movies = movies.drop('Unnamed: 3', axis=1, errors='ignore')
movies.head()


column_names1 = ['user_id','movie id','rating','timestamp']
ratings = pd.read_csv(csv_data_path,sep='::',header=None, names=column_names1, engine='python')
ratings.head()


user_cols = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
users = pd.read_csv(csv_file_path,sep='::', names=user_cols, engine='python')
users = users[['user_id', 'gender']]
users.head()


len(ratings), max(ratings['movie id']),min(ratings['movie id'])


merged_dataset = pd.merge(ratings, movies, how='inner', on='movie id')
merged_dataset.head()


merged_dataset[(merged_dataset['movie title'] == 'Chasing Amy (1997)') & (merged_dataset['user_id'] == 894)]


refined_dataset = merged_dataset.groupby(by=['user_id','movie title'], as_index=False).agg({"rating":"mean"})


refined_dataset.head()


# pivot and create movie-user matrix
user_to_movie_df = refined_dataset.pivot(
    index='user_id',
     columns='movie title',
      values='rating').fillna(0)


user_to_movie_df.head()
data = pd.merge(ratings, users, on='user_id')
data = pd.get_dummies(data, columns=['gender'])


# transform matrix to scipy sparse matrix
user_to_movie_sparse_df = csr_matrix(user_to_movie_df.values)
user_to_movie_sparse_df
     
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(user_to_movie_sparse_df)


## function to find top n similar users of the given input user
def get_similar_users(user):
    ## input to this function is the user.
   
    max_user_id = user_to_movie_df.index.max()
    knn_input = np.asarray([user_to_movie_df.values[user-1]]).reshape(1,-1)
    knn_input = user_to_movie_df.iloc[user-1,:].values.reshape(1,-1)
    distances, indices = knn_model.kneighbors(knn_input, n_neighbors=6)
   
    similar_users = indices.flatten()[1:]
    distance_list = distances.flatten()[1:]
   
    # If the range does not include all user IDs used in the get_similar_users() function,
    # adjust the range accordingly.
    if max(similar_users) > max_user_id:
        start_user_id = max_user_id - 5 if max_user_id >= 5 else 0
        end_user_id = max_user_id
        similar_users = np.array([i for i in range(start_user_id, end_user_id+1) if i != user])
   
    return similar_users, distance_list


similar_user_list, distance_list = get_similar_users(999)
     
similar_user_list, distance_list


weightage_list = distance_list/np.sum(distance_list)
weightage_list


mov_rtngs_sim_users = user_to_movie_df.values[similar_user_list]
mov_rtngs_sim_users


movies_list = user_to_movie_df.columns
movies_list


weightage_list = weightage_list[:,np.newaxis] + np.zeros(len(movies_list))
weightage_list.shape


new_rating_matrix = weightage_list*mov_rtngs_sim_users
mean_rating_list = new_rating_matrix.sum(axis =0)
mean_rating_list


def recommend_movies(user, n):
    knn_input = np.asarray([user_to_movie_df.values[user-1]]).reshape(1,-1)
    knn_input = user_to_movie_df.iloc[user-1,:].values.reshape(1,-1)
    distances, indices = knn_model.kneighbors(knn_input, n_neighbors=5)


    similar_user_list = indices.flatten()[1:]
    distance_list = distances.flatten()[1:]
    weightage_list = distance_list/np.sum(distance_list)


    mov_rtngs_sim_users = user_to_movie_df.values[similar_user_list]
    movies_list = user_to_movie_df.columns
    weightage_list = weightage_list[:,np.newaxis] + np.zeros(len(movies_list))
    new_rating_matrix = weightage_list*mov_rtngs_sim_users
    mean_rating_list = new_rating_matrix.sum(axis =0)
   
   
    return list(movies_list[np.argsort(mean_rating_list)[::-1][:n]])


Rec_mov_list = recommend_movies(1, 5)


# Separate users by gender
user_gen_male = users[users['gender'] == 'M']
user_gen_female = users[users['gender'] == 'F']


# Calculate average precision scores for male users
max_user_id_male = len(users)
precisions_male = []
for user_id in range(1, (max_user_id_male) + 1):
    user = users.loc[users['user_id'] == user_id]
    if (user['gender'] != 'M').any():
        continue
    rated_movies = [m for m in list(user_to_movie_df.columns) if user_to_movie_df.loc[user_id, m] > 0]
    if not rated_movies:
        continue
    recommended_movies = recommend_movies(user_id, n=5)
    if len(recommended_movies) > len(rated_movies):
        recommended_movies = recommended_movies[:len(rated_movies)]
    precision = precision_score(np.ones(len(recommended_movies)), np.in1d(recommended_movies, rated_movies))
    precisions_male.append(precision)
avg_precision_male = np.mean(precisions_male)


# Calculate average precision scores for female users
max_user_id_female = len(users)
precisions_female = []
for user_id in range(1, (max_user_id_female) + 1):
    user = users.loc[users['user_id'] == user_id]
    if (user['gender'] != 'F').any():
        continue
    rated_movies = [m for m in list(user_to_movie_df.columns) if user_to_movie_df.loc[user_id, m] > 0]
    if not rated_movies:
        continue
    recommended_movies = recommend_movies(user_id, n=5)
    if len(recommended_movies) > len(rated_movies):
        recommended_movies = recommended_movies[:len(rated_movies)]
    precision = precision_score(np.ones(len(recommended_movies)), np.in1d(recommended_movies, rated_movies))
    precisions_female.append(precision)
avg_precision_female = np.mean(precisions_female)


# Perform a two-sample t-test to check for statistical significance
t_statistic, p_value = ttest_ind(precisions_male, precisions_female, equal_var=False)


# Print the results
print(f"Average precision score for male users: {avg_precision_male:.8f}")
print(f"Average precision score for female users: {avg_precision_female:.8f}")
print("T-test statistic:", t_statistic)
print(f"P-value of the two-sample t-test: {p_value:.4f}")


if p_value < 0.05:
    print("There is a statistically significant difference in average precision between male and female users.")
else:
    print("There is no statistically significant difference in average precision between male and female users.")


def check_demographic_parity(recommendations, demographic_info, demographic_category):
   
    # Count the number of recommendations for each demographic group
    group_counts = {}
    total_count = 0
    for recommendation in recommendations:
        if recommendation in demographic_info:
            group = demographic_info[recommendation][demographic_category]
            group_counts[group] = group_counts.get(group, 0) + 1
            total_count += 1
   
    # Calculate the percentage of recommendations for each demographic group
    group_percentages = {}
    for group, count in group_counts.items():
        group_percentages[group] = count / total_count
   
    # Check if the percentage of recommendations for each demographic group is roughly the same
    threshold = 0.1  # set a threshold for acceptable difference in percentage
    for group, percentage in group_percentages.items():
        for other_group, other_percentage in group_percentages.items():
            if group != other_group:
                if abs(percentage - other_percentage) > threshold:
                    return False
   
    return True


dem_par_M = check_demographic_parity(Rec_mov_list, users, user_gen_male)
dem_par_F = check_demographic_parity(Rec_mov_list, users, user_gen_female)
print(dem_par_F)
print(dem_par_M)


max_user_id_male = len(users)
accuracy_male = []
for user_id in range(1, (max_user_id_male) + 1):
    user = users.loc[users['user_id'] == user_id]
    if (user['gender'] != 'M').any():
        continue
    rated_movies = [m for m in list(user_to_movie_df.columns) if user_to_movie_df.loc[user_id, m] > 0]
    if not rated_movies:
        continue
    recommended_movies = recommend_movies(user_id, n=5)
    if len(recommended_movies) > len(rated_movies):
        recommended_movies = recommended_movies[:len(rated_movies)]
    accuracy = accuracy_score(np.ones(len(recommended_movies)), np.in1d(recommended_movies, rated_movies))
    accuracy_male.append(accuracy)
avg_accuracy_male = np.mean(accuracy_male)


max_user_id_female = len(users)
accuracy_female = []
for user_id in range(1, (max_user_id_female) + 1):
    user = users.loc[users['user_id'] == user_id]
    if (user['gender'] != 'F').any():
        continue
    rated_movies = [m for m in list(user_to_movie_df.columns) if user_to_movie_df.loc[user_id, m] > 0]
    if not rated_movies:
        continue
    recommended_movies = recommend_movies(user_id, n=5)
    if len(recommended_movies) > len(rated_movies):
        recommended_movies = recommended_movies[:len(rated_movies)]
    accuracy = accuracy_score(np.ones(len(recommended_movies)), np.in1d(recommended_movies, rated_movies))
    accuracy_female.append(accuracy)
avg_accuracy_female = np.mean(accuracy_female)
accuracy_parity = avg_accuracy_female - avg_accuracy_male


print(avg_accuracy_male)
print(avg_accuracy_female)
print("Accuracy Parity: {:.4f}".format(accuracy_parity))
