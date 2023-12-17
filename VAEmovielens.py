import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from scipy.stats import ttest_ind




csv_data_path = r'C:\Users\linus\Desktop\Python\ml-1m\ratings.dat'
csv_file_path = r'C:\Users\linus\Desktop\Python\ml-1m\users.dat'


# Load data
data = pd.read_csv(csv_data_path, sep='::', header=None, names=['user_id', 'movie', 'rating', 'timestamp'], engine='python')
num_users = data['user_id'].max()
num_movies = data['movie'].max()


user_cols = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
users = pd.read_csv(csv_file_path,sep='::', names=user_cols, engine='python')
users = users[['user_id', 'gender']]
users.head()


merged_data = pd.merge(data, users, on='user_id')
# Separate users by gender
user_gen_male = users[users['gender'] == 'M']
user_gen_female = users[users['gender'] == 'F']


train_data, test_data = train_test_split(merged_data, test_size=0.2, random_state=42)


# Convert to sparse matrix
train_matrix = sp.coo_matrix((train_data['rating'].values, (train_data['user_id'].values, train_data['movie'].values)))
test_matrix = sp.coo_matrix((test_data['rating'].values, (test_data['user_id'].values, test_data['movie'].values)))


# Convert to PyTorch tensors
train_tensor = torch.FloatTensor(train_matrix.toarray())
test_tensor = torch.FloatTensor(test_matrix.toarray())


# Define model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_size, 1000)
        self.fc21 = nn.Linear(1000, 100)
        self.fc22 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 1000)
        self.fc4 = nn.Linear(1000, input_size)


    def encode(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc21(x)
        logvar = self.fc22(x)
        return mu, logvar


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


    def decode(self, z):
        z = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(z))


    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_size), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + sigma * KLD


# Set hyperparameters
input_size = 3953
sigma = 1


# Instantiate the model and optimizer
model = VAE()
optimizer = torch.optim.Adam(model.parameters())


# Train the model
num_epochs = 1
model.train()
for epoch in range(num_epochs):
    train_loss = 0
    for i in range(train_tensor.shape[0]):
        optimizer.zero_grad()
        input_data = train_tensor[i].unsqueeze(0)
        recon_data, mu, logvar = model(input_data)
        loss = loss_function(recon_data, input_data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print('Epoch {}: Loss: {:.4f}'.format(epoch+1, train_loss/len(train_tensor)))


    # Evaluate the model on the test set
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i in range(test_tensor.shape[0]):
            input_data = test_tensor[i].unsqueeze(0)
            recon_data, mu, logvar = model(input_data)
            loss = loss_function(recon_data, input_data, mu, logvar)
            test_loss += loss.item()


    test_loss /= test_tensor.shape[0]
    print('Epoch [{}/{}], Test Loss: {:.4f}'.format(epoch+1, num_epochs, test_loss))
    epoch +1


def compute_accuracy(model, data_tensor):
    model.eval()
    with torch.no_grad():
        num_correct = 0
        num_total = 0
        for i in range(data_tensor.shape[0]):
            input_data = data_tensor[i].unsqueeze(0)
            recon_data, mu, logvar = model(input_data)
            predicted_rating = (recon_data > 0.5).float()
            true_rating = input_data
            num_correct += (predicted_rating == true_rating).sum().item()
            num_total += input_data.numel()
    accuracy = num_correct / num_total
    return accuracy


missing_users_M = set(user_gen_male['user_id'].values) - set(test_data['user_id'].unique())
user_gen_male = user_gen_male[~user_gen_male['user_id'].isin(missing_users_M)]


missing_users_F = set(user_gen_female['user_id'].values) - set(test_data['user_id'].unique())
user_gen_female = user_gen_female[~user_gen_female['user_id'].isin(missing_users_F)]


# Compute the accuracy for male users
test_data = test_data.reset_index(drop=True)
test_data.drop_duplicates(subset=['user_id'], inplace=True)
male_data = test_tensor[np.intersect1d(test_data['user_id'].values, user_gen_male['user_id'].values, assume_unique=True), :].reshape(-1, test_tensor.shape[1])
male_accuracy = compute_accuracy(model, male_data)
print("Accuracy for male users:", male_accuracy)


# Compute the accuracy for female users
test_data = test_data.reset_index(drop=True)
test_data.drop_duplicates(subset=['user_id'], inplace=True)
female_data = test_tensor[np.intersect1d(test_data['user_id'].values, user_gen_female['user_id'].values, assume_unique=True), :].reshape(-1, test_tensor.shape[1])
female_accuracy = compute_accuracy(model, female_data)
print("Accuracy for female users:", female_accuracy)


# Compute the accuracy parity
accuracy_parity = abs(male_accuracy - female_accuracy)
print("Accuracy parity:", accuracy_parity)


def compute_precision(model, data_tensor):
    model.eval()
    with torch.no_grad():
        true_positives = 0
        false_positives = 0
        for i in range(data_tensor.shape[0]):
            input_data = data_tensor[i].unsqueeze(0)
            recon_data, mu, logvar = model(input_data)
            predicted_rating = (recon_data > 0.5).float()
            true_rating = input_data
            true_positives += ((predicted_rating == true_rating) & (true_rating == 1)).sum().item()
            false_positives += ((predicted_rating != true_rating) & (true_rating == 0)).sum().item()
    precision = true_positives / (true_positives + false_positives)
    return precision


test_data = test_data.reset_index(drop=True)
test_data.drop_duplicates(subset=['user_id'], inplace=True)
male_data = test_tensor[np.intersect1d(test_data['user_id'].values, user_gen_male['user_id'].values, assume_unique=True), :].reshape(-1, test_tensor.shape[1])
male_precision = compute_precision(model, male_data)
print("Precision for male users:", male_precision)


test_data = test_data.reset_index(drop=True)
test_data.drop_duplicates(subset=['user_id'], inplace=True)
female_data = test_tensor[np.intersect1d(test_data['user_id'].values, user_gen_female['user_id'].values, assume_unique=True), :].reshape(-1, test_tensor.shape[1])
female_precision = compute_precision(model, female_data)
print("Precision for female users:", female_precision)


t_stat, p_value = ttest_ind(male_precision, female_precision, equal_var=False)
print("t-statistic:", t_stat)
print("p-value:", p_value)


if p_value < 0.05:
    print("The difference in precision between male and female users is statistically significant.")
else:
    print("The difference in precision between male and female users is not statistically significant.")


def calculate_demographic_parity(model, data_tensor, users):
    model.eval()
    with torch.no_grad():
        num_recommendations = torch.zeros((2,))
        num_total = torch.zeros((2,))
        for i, input_user in enumerate(users):
            input_data = data_tensor[i].unsqueeze(0)
            input_user = users[i]
            recon_data, mu, logvar = model(input_data)
            predicted_rating = (recon_data > 0.5).float()
            true_rating = input_data
            if input_user == 0: # Male user
                num_total[0] += 1
                num_recommendations[0] += predicted_rating.sum()
            else: # Female user
                num_total[1] += 1
                num_recommendations[1] += predicted_rating.sum()




    percentage_recommendations = num_recommendations / num_total
    percentage_users = torch.tensor([len(user_gen_male) / num_users, len(user_gen_female) / num_users])
    demographic_parity_diff = percentage_recommendations - percentage_users
    return percentage_recommendations, percentage_users, demographic_parity_diff


users = torch.tensor([int(users.loc[test_data.iloc[i]['user_id']-1]['gender']=='M') for i in range(test_data.shape[0])])
percentage_recommendations, percentage_users, demographic_parity_diff = calculate_demographic_parity(model, test_tensor, users)
print('Percentage of recommendations for male users: {:.2f}%'.format(percentage_recommendations[0] * 100))
print('Percentage of recommendations for female users: {:.2f}%'.format(percentage_recommendations[1] * 100))
print('Percentage of male users: {:.2f}%'.format(percentage_users[0] * 100))
print('Percentage of female users: {:.2f}%'.format(percentage_users[1] * 100))
print('Demographic parity difference: {:.2f}%'.format(demographic_parity_diff.abs().sum() * 100))
