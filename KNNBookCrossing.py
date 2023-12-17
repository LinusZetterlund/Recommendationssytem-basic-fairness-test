import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from scipy.stats import ttest_ind
from sklearn.metrics import precision_score, accuracy_score


csv_data_path = r'C:\Users\linus\Desktop\Python\bookfodder\BX-Book-Ratings.csv'
csv_item_path = r'C:\Users\linus\Desktop\Python\bookfodder\BX-Books.csv'
csv_file_path = r'C:\Users\linus\Desktop\Python\bookfodder\BX-Users.csv'


d = 'ISBN | Book-Title | Book-Author | Year-Of-Publication | Publisher | Image-URL-S | Image-URL-M | Image-URL-L '
column_names2 = d.split(' | ')
column_names2


movies = pd.read_csv(csv_item_path, delimiter=';', encoding='latin-1', engine='python', header=None, names=column_names2, error_bad_lines=False)
movies = movies.drop('Unnamed: 3', axis=1, errors='ignore')
movies.head()


column_names1 = ['user_id','ISBN','Book-Rating']
ratings = pd.read_csv(csv_data_path,sep=';',header=None, names=column_names1, engine='python', encoding='latin-1', error_bad_lines=False)
ratings.head()


user_cols = ['user_id', 'Location', 'Age',]
users = pd.read_csv(csv_file_path, sep=';', names=user_cols, engine='python', encoding='latin-1', error_bad_lines=False, skiprows=1)
users = users[['user_id', 'Age']]
users.head()




# Convert 'Age' column to numeric
users['Age'] = pd.to_numeric(users['Age'], errors='coerce')




ratings['ISBN'] = ratings['ISBN'].astype(str)
merged_dataset = pd.merge(ratings, movies, on='ISBN', how='inner')
merged_dataset.head()
merged_dataset['Book-Rating'] = pd.to_numeric(merged_dataset['Book-Rating'], errors='coerce')


refined_dataset = merged_dataset.groupby(by=['user_id','ISBN'], as_index=False).agg({"Book-Rating":"mean"})
print(refined_dataset.shape)
# Set the minimum number of ratings a book must have
min_ratings = 20


# Group the dataset by ISBN and count the number of ratings for each book
ratings_per_book = refined_dataset.groupby('ISBN')['Book-Rating'].count()


# Keep only the ISBNs with more than min_ratings ratings
valid_books = ratings_per_book[ratings_per_book >= min_ratings].index.tolist()


# Filter the dataset to keep only the valid books
refined_dataset = refined_dataset[refined_dataset['ISBN'].isin(valid_books)]
refined_dataset.head()


print("Number of unique users:", len(refined_dataset['user_id'].unique()))
print("Number of unique books:", len(refined_dataset['ISBN'].unique()))
# pivot and create movie-user matrix
user_to_movie_df = refined_dataset.pivot(
    index='user_id',
     columns='ISBN',
      values='Book-Rating').fillna(0)


user_to_movie_df.head()








# transform matrix to scipy sparse matrix
user_to_movie_sparse_df = csr_matrix(user_to_movie_df.values)
user_to_movie_sparse_df
     
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(user_to_movie_sparse_df)


## function to find top n similar users of the given input user
def get_similar_users(user):
    max_user_id = int(user_to_movie_df.index.max())
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


# Separate users by age
users_below_30 = users[users['Age'] <= 30]
users_above_30 = users[users['Age'] > 30]


if 'ISBN' in user_to_movie_df.columns:
    user_to_movie_df['ISBN'] = user_to_movie_df['ISBN'].astype(int)


max_user_id_below_30 = int(users_below_30['user_id'].max())
precisions_below_30 = []
for user_id in range(0, max_user_id_below_30 + 1):
    user = users.loc[users['user_id'] == user_id]
    if not user.empty and user['Age'].iloc[0] > 30:
        continue
    user_id_int = int(user_id)
    if user_id_int >= len(user_to_movie_df):
        print("Error: user_id_int is out of range")
        break
    else:
        if user_to_movie_df.iloc[user_id_int].sum() == 0:
            continue
    recommended_movies = recommend_movies(user_id, n=1)
    if len(recommended_movies) > user_to_movie_df.iloc[user_id_int].sum():
        recommended_movies = recommended_movies[:int(user_to_movie_df.iloc[user_id_int].sum())]
    precision = precision_score(np.ones(len(recommended_movies)), np.in1d(recommended_movies, user_to_movie_df.columns))
    precisions_below_30.append(precision)
    if user_id_int >= len(user_to_movie_df):
        print("Error: user_id_int is out of range")
        break
avg_precision_below_30 = np.mean(precisions_below_30)




# Calculate average precision scores for female users
max_user_id_above_30 = int(users_above_30['user_id'].max())
precisions_above_30 = []
for user_id in range(0, max_user_id_above_30 + 1):
    user = users.loc[users['user_id'] == user_id]
    if not user.empty and user['Age'].iloc[0] > 30:
        continue
    user_id_int = int(user_id)
    if user_id_int >= len(user_to_movie_df):
        print("Error: user_id_int is out of range")
        break
    else:
        if user_to_movie_df.iloc[user_id_int].sum() == 0:
            continue
    recommended_movies = recommend_movies(user_id, n=1)
    if len(recommended_movies) > user_to_movie_df.iloc[user_id_int].sum():
        recommended_movies = recommended_movies[:int(user_to_movie_df.iloc[user_id_int].sum())]
    precision = precision_score(np.ones(len(recommended_movies)), np.in1d(recommended_movies, user_to_movie_df.columns))
    precisions_above_30.append(precision)
    if user_id_int >= len(user_to_movie_df):
        print("Error: user_id_int is out of range")
        break
avg_precision_above_30 = np.mean(precisions_above_30)




# Perform a two-sample t-test to check for statistical significance
t_statistic, p_value = ttest_ind(precisions_below_30, precisions_above_30, equal_var=False)


# Print the results




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


dem_par_M = check_demographic_parity(Rec_mov_list, users, users_below_30)
dem_par_F = check_demographic_parity(Rec_mov_list, users, users_above_30)




max_user_id_above_30 = int(users_above_30['user_id'].max())
accuracy_below_30 = []
for user_id in range(0, max_user_id_above_30 + 1):
    user = users.loc[users['user_id'] == user_id]
    if not user.empty and user['Age'].iloc[0] > 30:
        continue
    user_id_int = int(user_id)
    if user_id_int >= len(user_to_movie_df):
        print("Error: user_id_int is out of range")
        break
    else:
        if user_to_movie_df.iloc[user_id_int].sum() == 0:
            continue
    recommended_movies = recommend_movies(user_id, n=1)
    if len(recommended_movies) > user_to_movie_df.iloc[user_id_int].sum():
        recommended_movies = recommended_movies[:int(user_to_movie_df.iloc[user_id_int].sum())]
    accuracy = accuracy_score(np.ones(len(recommended_movies)), np.in1d(recommended_movies, user_to_movie_df.columns))
    accuracy_below_30.append(accuracy)
    if user_id_int >= len(user_to_movie_df):
        print("Error: user_id_int is out of range")
        break
avg_accuracy_below_30 = np.mean(accuracy_below_30)


# Calculate average precision scores for female users
max_user_id_above_30 = int(users_above_30['user_id'].max())
accuracy_above_30 = []
for user_id in range(0, max_user_id_above_30 + 1):
    user = users.loc[users['user_id'] == user_id]
    if not user.empty and user['Age'].iloc[0] > 30:
        continue
    user_id_int = int(user_id)
    if user_id_int >= len(user_to_movie_df):
        print("Error: user_id_int is out of range")
        break
    else:
        if user_to_movie_df.iloc[user_id_int].sum() == 0:
            continue
    recommended_movies = recommend_movies(user_id, n=1)
    if len(recommended_movies) > user_to_movie_df.iloc[user_id_int].sum():
        recommended_movies = recommended_movies[:int(user_to_movie_df.iloc[user_id_int].sum())]
    accuracy = accuracy_score(np.ones(len(recommended_movies)), np.in1d(recommended_movies, user_to_movie_df.columns))
    accuracy_above_30.append(accuracy)
    if user_id_int >= len(user_to_movie_df):
        print("Error: user_id_int is out of range")
        break
avg_accuracy_above_30 = np.mean(accuracy_above_30)




accuracy_parity = avg_precision_above_30 - avg_accuracy_below_30




print(avg_accuracy_below_30)
print(avg_accuracy_above_30)
print("Accuracy Parity: {:.4f}".format(accuracy_parity))


print(dem_par_F)
print(dem_par_M)


print(f"Average precision score for male users: {avg_precision_below_30:.8f}")
print(f"Average precision score for female users: {avg_precision_above_30:.8f}")
print("T-test statistic:", t_statistic)
print(f"P-value of the two-sample t-test: {p_value:.4f}")






if p_value < 0.05:
    print("There is a statistically significant difference in average precision between male and female users.")
else:
    print("There is no statistically significant difference in average precision between male and female users.")
