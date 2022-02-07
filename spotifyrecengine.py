import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import re

pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

client_credentials_manager = SpotifyClientCredentials('eda08d5ff18641cbaeece95bf66843b0',
                                                      'f7f41edb58fd4c0bbb13f4819cdc988a')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Dataset DF & Tracks DF
complete_feature_dataframe = pd.read_csv('SpotifyFeatures.csv')
tracks = complete_feature_dataframe[['track_name', 'track_id']].copy()

URL = "https://open.spotify.com/playlist/37i9dQZF1DX4o1oenSJRJd?si=953a8ad2e54f42c1"

# One Hot Encoding the categorical columns of dataset
genre_OHE = pd.get_dummies(complete_feature_dataframe.genre)
key_OHE = pd.get_dummies(complete_feature_dataframe.key)

# Dropping the categorical columns and the ones that are not features of the songs
complete_feature_dataframe = complete_feature_dataframe.drop('genre', axis=1)
complete_feature_dataframe = complete_feature_dataframe.drop('artist_name', axis=1)
complete_feature_dataframe = complete_feature_dataframe.drop('track_name', axis=1)
complete_feature_dataframe = complete_feature_dataframe.drop('popularity', axis=1)
complete_feature_dataframe = complete_feature_dataframe.drop('key', axis=1)
complete_feature_dataframe = complete_feature_dataframe.drop('mode', axis=1)
complete_feature_dataframe = complete_feature_dataframe.drop('time_signature', axis=1)

# joining the one hot encoded columns from categorical columns i.e, genre and key
complete_feature_dataframe = complete_feature_dataframe.join(genre_OHE)
complete_feature_dataframe = complete_feature_dataframe.join(key_OHE)

# Function to extract all the songs in the playlist from URL
# and after creating a dataframe from it, we are going to return it

def getPlaylistDataFrame(URL):
    # Extracting the playlist_id after we split the URL by '/' and using the 
    # String and 4th index. Afterwards in that index, extracting all alphabets before
    # '?'
    playlist_id = URL.split("/")[4].split("?")[0]

    # Extracting playlist dictionary from spotify by having playlist_id
    playlist_tracks_data = sp.playlist_tracks(playlist_id)

    # Creating lists to extract information from playlist
    playlist_tracks_id = []
    playlist_tracks_titles = []
    playlist_tracks_artists = []
    playlist_tracks_first_artists = []
    playlist_tracks_genres = []

    # Extracting the ids, titles, artists, and first artists, genres from the paylist and appending
    # The values one by one into the lists
    for track in playlist_tracks_data['items']:
        playlist_tracks_id.append(track['track']['id'])
        playlist_tracks_titles.append(track['track']['name'])

        artist_list = []
        for artist in track['track']['artists']:
            artist_list.append(artist['name'])
        playlist_tracks_artists.append(artist_list)
        playlist_tracks_first_artists.append(artist_list[0])

        artist_uri = track["track"]["artists"][0]["uri"]
        artist_info = sp.artist(artist_uri)
        artist_genres = artist_info["genres"]

        genre_list = []
        for genre in artist_genres:
            genre_list.append(genre)

        playlist_tracks_genres.append(genre_list)

    # Extracting the audio features from spotify by using playlist_tracks_ids
    features = sp.audio_features(playlist_tracks_id)

    # Creating features dataframe from features list
    features_df = pd.DataFrame(data=features, columns=features[0].keys())

    # Defining the title, first_artist, all artistsm genre for feature_df by using the already created lists
    features_df['title'] = playlist_tracks_titles
    features_df['first_artist'] = playlist_tracks_first_artists
    features_df['all_artists'] = playlist_tracks_artists
    features_df['genre'] = playlist_tracks_genres

    # Extracting the required columns
    features_df = features_df[['id', 'title', 'first_artist', 'all_artists',
                               'acousticness', 'danceability', 'duration_ms',
                               'energy', 'instrumentalness', 'liveness',
                               'loudness', 'speechiness', 'tempo', 'valence', 'key', 'genre']]

    return features_df


playlist_dataframe = getPlaylistDataFrame(URL)

# These are the feature columns from the dataset from spotifyFeatures.csv
# from A cappella to Rock are genres and from A to G# are keys

feature_columns = ['A Capella', 'Alternative', 'Anime', 'Blues',
                   "Children's Music", 'Children’s Music', 'Classical', 'Comedy',
                   'Country', 'Dance', 'Electronic', 'Folk', 'Hip-Hop', 'Indie', 'Jazz',
                   'Movie', 'Opera', 'Pop', 'R&B', 'Rap', 'Reggae', 'Reggaeton', 'Rock',
                   'Ska', 'Soul', 'Soundtrack', 'World', 'A', 'A#', 'B', 'C', 'C#', 'D',
                   'D#', 'E', 'F', 'F#', 'G', 'G#']

# going to append all the columns from play_list_dataframe so that we keep track of all the features
for index in playlist_dataframe.index:
    col_name = playlist_dataframe.iloc[index]['genre']
    for word in col_name:
        if word.title() not in feature_columns:
            feature_columns.append(word.title())


# This function will return one hot encoded features from the playlist passed to it with categorical genre and key
# columns
def create_playlist_df(playlist):
    playlist_df = playlist.copy()

    # The keys are mapping to the index values. The index values are actually the pitch values ranging from 0-11
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # creating new columns for each genre and keys
    for column in feature_columns:
        playlist_df[column] = 0.0

    # one hot encoding the genre and key columns
    for index in playlist_df.index:
        genres = playlist_df.iloc[index]["genre"]
        for genre in genres:
            playlist_df.at[index, genre.title()] = 1

        key_index = playlist_df.iloc[index]["key"]
        if 0 <= key_index <= 11:
            playlist_df.at[index, keys[key_index]] = 1

    # going to drop the categorical columns and those which are not the features of the songs    
    playlist_df = playlist_df.drop('genre', axis=1)
    playlist_df = playlist_df.drop('key', axis=1)
    playlist_df = playlist_df.drop('title', axis=1)
    playlist_df = playlist_df.drop('first_artist', axis=1)
    playlist_df = playlist_df.drop('all_artists', axis=1)

    return playlist_df


# created playlist features dataframe, which is essentially only the features of each track
playlist_features_df = create_playlist_df(playlist_dataframe)

# going to add the columns extracted from playlist_features_df so that both the dataframes are
# campatible and have the same shapes and columns for the cosine functionality
for column in feature_columns:
    if column not in complete_feature_dataframe.columns.values:
        complete_feature_dataframe[column] = 0

# storing the playlist_features columns and columns from complete_features_df so that they can
# be used later to extract specific features
playlist_features_df_columns = playlist_features_df.columns.values
complete_feature_dataframe_columns = complete_feature_dataframe.columns.values

# Normalization
# Applying scaling in the original complete feature dataframe
scalar = MinMaxScaler()
scalar.fit(complete_feature_dataframe[complete_feature_dataframe_columns[1:]])
scaled_features = scalar.transform(complete_feature_dataframe[complete_feature_dataframe_columns[1:]])

# scaled_features returned from the transform function was an array, so we have to create the dataframe
# from it again
scaled_features = pd.DataFrame(scaled_features, columns=complete_feature_dataframe_columns[1:])

scaled_features['track_id'] = complete_feature_dataframe['track_id']
complete_feature_dataframe = scaled_features.copy()

# Going to apply the same scaling we applied on the complete_playlist dataframe on the playlist_features_df
scaled_features = scalar.transform(playlist_features_df[playlist_features_df_columns[1:]])
scaled_features = pd.DataFrame(scaled_features, columns=playlist_features_df_columns[1:])

scaled_features['id'] = playlist_features_df['id']
playlist_features_df = scaled_features.copy()
playlist_features_df

complete_feature_dataframe


# Generate Recommended
# Playlist Returns top 5 similar songs, dataframe consists of only tracks, id's and only returns
# top cosine similarity results from the dataset.

def generate_recommend_playlist(feature_df, playlist_df, df):
    # Copying the playlist dataframe columns to make sure they are the same as the dataset.
    complete_feature_set_playlist = playlist_df[playlist_features_df_columns].copy()

    # The songs which are not in the current playlist will be saved to complete_feature_set_nonplaylist which will be
    # used to return recommendations.
    complete_feature_set_nonplaylist = feature_df[~feature_df['track_id'].isin(playlist_df['id'].values)]

    # Saving the dataframe with playlist except for the tracks which are alread in it, because they would have a
    # cosine similarity of 1.
    df = df[~df['track_id'].isin(playlist_df['id'].values)]

    # Dropping id column as it is not needed for cosine similarity.
    complete_feature_set_playlist_final = complete_feature_set_playlist.drop(columns="id")

    # This sum will be used later in cosine similarity function to find similarity between recommended songs and the
    # current playlist.
    features = complete_feature_set_playlist_final.sum(axis=0)

    # saving a copy of complete_feature_set_nonplaylist to non_playlist_features 
    nonplaylist_features = complete_feature_set_nonplaylist[complete_feature_dataframe_columns].copy()

    # Creating an array from features dataset
    features_array = features.values

    # Reshaping features array to a two-dimensional array to the new shape which will be (1, number of values in
    # features_array)
    features_two_dimensional_array = features_array.reshape(1, -1)

    # Not going to use id as a feature for recommendation array
    non_playlist_features_without_id = nonplaylist_features.drop('track_id', axis=1)

    # Going to save the ids in a single array
    non_playlist_features_without_id_array = non_playlist_features_without_id.values

    # Using cosine similarity to calculate between all the songs outside current playlist and the features from
    # the current playlist. This will return a two-dimensional array having a single vector of cosine values
    two_dimensional_array = cosine_similarity(non_playlist_features_without_id_array, features_two_dimensional_array)

    # Going to vectorize the two-dimensional array so that it can be stored in a specific column afterwards
    single_array = two_dimensional_array[:, 0]

    # Storing the cosine values in a separate column. The values correspond to the respective tracks to show how much
    # similarity they have. The larger the value the greater is the similarity and hence good for recommendation.
    df['sim'] = single_array

    # Sorting the dataframe based on the cosine similarity calculated. Those songs which have highest similarity will
    # be on top.
    non_playlist_df_top_5 = df.sort_values('sim', ascending=False).head(5)

    return non_playlist_df_top_5


result = generate_recommend_playlist(complete_feature_dataframe, playlist_features_df, tracks)

# Printing the top five songs 
for i in range(len(result)):
    print('\n'+str(result.iloc[i, 0]) + ' - ' + "https://open.spotify.com/track/" + str(result.iloc[i, 1]).split("/")[-1])
