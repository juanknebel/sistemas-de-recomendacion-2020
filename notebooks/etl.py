import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse


def add_mean_by_movie(ratings, ratings_test):
    df_movie_mean = ratings.groupby("movieID")["rating"].mean().reset_index()
    df_movie_mean.rename(columns={"rating": "mean_rating"}, inplace=True)
    global_mean = ratings["rating"].mean()

    ratings = ratings.merge(
        df_movie_mean, how="left", left_on="movieID", right_on="movieID"
    )
    ratings_test = ratings_test.merge(
        df_movie_mean, how="left", left_on="movieID", right_on="movieID"
    )
    ratings_test.fillna(global_mean, inplace=True)
    ratings_test.rating = None

    return ratings, ratings_test


def generate_profile_by_user(ratings, genres):
    list_of_genres = genres.genre.unique().tolist()
    list_of_genres.sort()
    genres["value"] = 1
    df_genre_matrix = (
        genres.pivot(index="movieID", columns="genre", values="value")
        .fillna(0)
        .rename_axis(columns=None)
        .reset_index()
    )

    rating_genre = df_genre_matrix.merge(
        ratings, left_on="movieID", right_on="movieID"
    )

    list_of_profiles = []
    print("Generating profiling by user ...")
    for a_user in tqdm(ratings.userID.unique()):
        profile_by_user = dict()
        profile_by_user["userID"] = a_user
        df_user = rating_genre[rating_genre.userID == a_user]

        for a_genre in list_of_genres:
            profile_by_user[a_genre] = (
                df_user.loc[df_user[a_genre] == 1, "rating"].mean() / 5.0
            )

        list_of_profiles.append(profile_by_user)

    return pd.DataFrame.from_dict(list_of_profiles).fillna(0)


def add_similarity_by_movie(ratings, ratings_test):
    genres = pd.read_csv("./data/movie_genres.csv")
    movies = pd.read_csv("./data/movies.csv")
    profile_by_user = generate_profile_by_user(ratings, genres)

    list_of_genres = genres.genre.unique().tolist()
    list_of_genres.sort()
    genres["value"] = 1
    genres = (
        genres.pivot(index="movieID", columns="genre", values="value")
        .fillna(0)
        .rename_axis(columns=None)
        .reset_index()
    )
    movies = movies.merge(genres, left_on="id", right_on="movieID")

    print("Similarity by each review based on the profile in train")
    list_of_user_movie_similarity = []
    for user_id, group in tqdm(ratings.groupby("userID")):
        for _, row in group.iterrows():
            user_movie_similarity = dict()
            profile = profile_by_user.loc[profile_by_user.userID == user_id][
                list_of_genres
            ].values[0]
            movie = movies.query("id == {}".format(row.movieID))[
                list_of_genres
            ].values[0]
            similarity = np.dot(movie, profile) / (
                np.linalg.norm(movie) * np.linalg.norm(profile)
            )
            user_movie_similarity["userID"] = user_id
            user_movie_similarity["movieID"] = row.movieID
            user_movie_similarity["similarity"] = similarity
            list_of_user_movie_similarity.append(user_movie_similarity)
    rating_similarity = pd.DataFrame(list_of_user_movie_similarity)
    ratings = ratings.merge(
        rating_similarity,
        left_on=["userID", "movieID"],
        right_on=["userID", "movieID"],
    )

    print("Similarity by each review based on the profile in test")
    list_of_user_movie_similarity = []
    for user_id, group in tqdm(ratings_test.groupby("userID")):
        for _, row in group.iterrows():
            user_movie_similarity = dict()
            profile = profile_by_user.loc[profile_by_user.userID == user_id][
                list_of_genres
            ].values[0]
            movie = movies.query("id == {}".format(row.movieID))[
                list_of_genres
            ].values[0]
            similarity = np.dot(movie, profile) / (
                np.linalg.norm(movie) * np.linalg.norm(profile)
            )
            user_movie_similarity["userID"] = user_id
            user_movie_similarity["movieID"] = row.movieID
            user_movie_similarity["similarity"] = similarity
            list_of_user_movie_similarity.append(user_movie_similarity)
    rating_similarity_test = pd.DataFrame(list_of_user_movie_similarity)
    ratings_test = ratings_test.merge(
        rating_similarity_test,
        left_on=["userID", "movieID"],
        right_on=["userID", "movieID"],
    )

    return ratings, ratings_test


def save_to_csv(data, file_name):
    data.to_csv(file_name, index=False)


def execute_all():
    df_ratings = pd.read_csv("./data/ratings_train.csv")
    df_ratings_test = pd.read_csv("./data/ratings_test.csv")

    train, test = add_mean_by_movie(df_ratings, df_ratings_test)
    train, test = add_similarity_by_movie(df_ratings, df_ratings_test)

    save_to_csv(train, "./data/ratings_train_mean_similarity.csv")
    save_to_csv(test, "./data/ratings_test_mean_similarity.csv")


if __name__ == "__main__":
    pipeline_executions = {
        1: [add_mean_by_movie],
        2: [add_similarity_by_movie],
    }

    execution_help = ""
    for k, v in pipeline_executions.items():
        execution_help = (
            execution_help
            + f"Step {k} execute {', '.join([m.__name__ for m in v])}.\n"
        )
    parser = argparse.ArgumentParser(description="ETL cli.")
    parser.add_argument(
        "-s",
        "--step",
        dest="step",
        type=int,
        nargs=None,
        help=f"which step of the etl execute. {execution_help}",
    )
    parser.add_argument(
        "-p",
        "--postfix",
        dest="postfix",
        type=int,
        nargs=None,
        help=f"number to indicate the postfix in the outputfiles",
        required=True,
    )
    parser.add_argument(
        "-a",
        "--all",
        dest="execute_all",
        action="store_const",
        const="all",
        help="execute the entire etl",
    )
    args = parser.parse_args()

    if args.execute_all:
        execute_all()
    elif args.step:
        train = pd.read_csv("./data/ratings_train.csv")
        test = pd.read_csv("./data/ratings_test.csv")
        for a_method in pipeline_executions[args.step]:
            train, test = a_method(train, test)
            save_to_csv(train, f"./data/ratings_train_{args.postfix}.csv")
            save_to_csv(test, f"./data/ratings_test_{args.postfix}.csv")
