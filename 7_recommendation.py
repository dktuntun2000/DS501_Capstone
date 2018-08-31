#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 15:44:08 2018

@author: wenzhiwang
"""

import pandas as pd

import numpy as np

import os

os.getcwd()


# read play, download, search
df_play = pd.read_csv('play_ds.csv')

df_play.head()

df_download = pd.read_csv('down_ds.csv')


df_download['download'] = 1

# I don't look at different time windows for simplicity

df_download.drop(['device','date'],axis = 1, inplace = True)

df_down = df_download.groupby(['uid','song_id']).agg('sum')

df_down.reset_index(inplace = True)

# filter out records with negative time

df_play['play_time'] = pd.to_numeric(df_play.play_time, errors='coerce')

df_play = df_play[ (df_play.play_time >= 0) & ( df_play.song_length >0) ]

df_play.dropna(inplace = True)

df_play.drop(['device','date'],axis = 1,inplace = True)

df_play_u_song = df_play.groupby(['uid','song_id']).agg('sum')

df_play_u_song.reset_index(inplace = True)


df_play_u_song['play'] = df_play_u_song.play_time/df_play_u_song.song_length

df_play_u_song = df_play_u_song[df_play_u_song.play <= 1]


# Merge download and play dataset
df = pd.merge(df_play_u_song,df_down, on = ['uid','song_id'], how = 'outer')

df.drop(['play_time','song_length'], axis = 1, inplace = True)


# Compute implicit review score
df['score'] = 0

for i in range(len(df.play)):
    
    if df.download[i] >= 5:
        df['score'][i] = 7
    elif df.download[i] < 5 and df.download[i] >= 1 and df.play[i] > 0 :
        df['score'][i] = 6
    elif df.download[i] < 5 and df.download[i] >= 1 and np.isnan(df.play[i]):
        df['score'][i] = 5
    elif np.isnan(df.download[i]) and df.play[i] >= 0.8:
        df['score'][i] = 4
    elif np.isnan(df.download[i]) and df.play[i] >= 0.6 and df.play[i] < 0.8 :
        df['score'][i] = 3
    elif np.isnan(df.download[i]) and df.play[i] >= 0.3 and df.play[i] < 0.6 :
        df['score'][i] = 2
    else:
        df['score'][i] = 1
        
df.drop(['play','download'], axis = 1, inplace = True)

df_utility = pd.pivot_table(data=df, 
                            values='score', 
                            index='uid', 
                            columns='song_id', 
                            fill_value=0)
    

# Item-Item Similarity Matrix
item_sim_mat = cosine_similarity(df_utility.T)

least_to_most_sim_indexes = np.argsort(item_sim_mat, axis=1)

# Neighborhoods
neighborhood_size = 75
neighborhoods = least_to_most_sim_indexes[:, -neighborhood_size:]

# Let's pick a lucky user
user_id = df_utility.index[100]

n_users = df_utility.shape[0]
n_items = df_utility.shape[1]

start_time = time()

items_rated_by_this_user = df_utility.loc[user_id].nonzero()[0]



# Just initializing so we have somewhere to put rating preds
out = np.zeros(n_items)
for item_to_rate in range(n_items):
    relevant_items = np.intersect1d(neighborhoods[item_to_rate],
                                    items_rated_by_this_user,
                                    assume_unique=True)  # assume_unique speeds up intersection op

    out[item_to_rate] = np.dot(df_utility.loc[user_id][list(relevant_items)].values ,\
        item_sim_mat[item_to_rate, list(relevant_items)] )/ \
        item_sim_mat[item_to_rate, list(relevant_items)].sum()


pred_ratings = np.nan_to_num(out)

print(pred_ratings)

print("Execution time: %f seconds" % (time()-start_time))



# Recommend n business
n = 10

# Get item indexes sorted by predicted rating
item_index_sorted_by_pred_rating = list(np.argsort(pred_ratings))[::-1]

# Find items that have been rated by user
items_rated_by_this_user = df_utility.loc[user_id].nonzero()[0]

# We want to exclude the items that have been rated by user
unrated_items_by_pred_rating = [item for item in item_index_sorted_by_pred_rating
                                if item not in items_rated_by_this_user]

unrated_items_by_pred_rating[:n]



## NFM and UVC 

ratings_mat = np.mat(df_utility.values)



from sklearn.decomposition import NMF

def fit_nmf(M,k):
    nmf = NMF(n_components=k)
    nmf.fit(M)
    W = nmf.transform(M);
    H = nmf.components_;
    err = nmf.reconstruction_err_
    return W,H,err

# decompose
W,H,err = fit_nmf(ratings_mat,200)

print(err)

print(W.shape,H.shape)

# reconstruct
ratings_mat_fitted = W.dot(H)
errs = np.array((ratings_mat-ratings_mat_fitted).flatten()).squeeze()
mask = np.array(ratings_mat.flatten()).squeeze()>0

mse = np.mean(errs[mask]**2)
average_abs_err = abs(errs[mask]).mean()
print(mse)
print(average_abs_err)



# get recommendations for one user
# user_id = df_utility.index[100]

user_id = 100
n = 10

pred_ratings = ratings_mat_fitted[user_id,:]
item_index_sorted_by_pred_rating = list(np.argsort(pred_ratings))[::-1]

items_rated_by_this_user = ratings_mat[user_id].nonzero()[1]

unrated_items_by_pred_rating = [item for item in item_index_sorted_by_pred_rating
                                if item not in items_rated_by_this_user]

unrated_items_by_pred_rating[:n]



### check errors
# truth
ratings_true = ratings_mat[user_id, items_rated_by_this_user]
# prediction
ratings_pred = pred_ratings[items_rated_by_this_user]
print(list(zip(np.array(ratings_true).squeeze(),ratings_pred)))


err_one_user = ratings_true-ratings_pred


print(err_one_user)


print(abs(err_one_user).mean())





######## Let's try UVD

from sklearn.decomposition import TruncatedSVD

def fit_uvd(M,k):
    # use TruncatedSVD to realize UVD
    svd = TruncatedSVD(n_components=k, n_iter=7, random_state=0)
    svd.fit(M)

    V = svd.components_
    U = svd.transform(M) # effectively, it's doing: U = M.dot(V.T)
    # we can ignore svd.singular_values_ for our purpose
    
    # why we can do this?
    # recall: 
    # SVD start from u*s*v=M => u*s=M*v.T, where M*v.T is our transformation above 
    # to get U in UVD
    # so the above U is effectively u*s in SVD
    # that's why U*V = u*s*v = M our original matrix
    # there are many ways to understand it!
    # here we by-passed singular values.
    
    return U,V, svd

# decompose
U,V,svd = fit_uvd(ratings_mat,200)


print(U.shape,V.shape)


# reconstruct
ratings_mat_fitted = U.dot(V) # U*V


# recall: U = M.dot(V.T), then this is M.dot(V.T).dot(V)
# original M is transformed to new space, then transformed back
# this is another way to understand it!

# calculate errs
errs = np.array((ratings_mat-ratings_mat_fitted).flatten()).squeeze()
mask = np.array((ratings_mat).flatten()).squeeze()>0

mse = np.mean(errs[mask]**2)
average_abs_err = abs(errs[mask]).mean()
print(mse)
print(average_abs_err)



# compare with another way to reconstruct matrix
# with the above "tranformed to the new space and back" language
# without the UV language, we can do:

# reconstruct M with inverse_transform
ratings_mat_fitted_2 = svd.inverse_transform(svd.transform(ratings_mat))
ratings_mat_fitted = U.dot(V)
print(sum(sum(ratings_mat_fitted - ratings_mat_fitted_2)))
# they are just equivalent!!


# get recommendations for one user
user_id = 100
n = 10

pred_ratings = ratings_mat_fitted[user_id,:]
item_index_sorted_by_pred_rating = list(np.argsort(pred_ratings))[::-1]

items_rated_by_this_user = ratings_mat[user_id].nonzero()[1]

unrated_items_by_pred_rating = [item for item in item_index_sorted_by_pred_rating
                                if item not in items_rated_by_this_user]

unrated_items_by_pred_rating[:n]



### check errors
# truth
ratings_true = ratings_mat[user_id, items_rated_by_this_user]
# prediction
ratings_pred = pred_ratings[items_rated_by_this_user]
print(list(zip(np.array(ratings_true).squeeze(),ratings_pred)))



err_one_user = ratings_true-ratings_pred
print(err_one_user)

print(abs(err_one_user).mean())






    
    













