from flask import Flask, request, render_template, redirect
import os
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.functions import lit
from datetime import datetime

class MovieManager:
    def __init__(self):
        def parseImdb(line):
            fields = line.value.split(',')
            return Row(movieID = int(fields[0]), ImdbID = fields[1])

        def parseAbouts(line):
            fields = line.value.split(',')
            return Row(movieID = int(fields[0]), movieNameCat = (fields[1], fields[2].replace("|", ", ")))
        
        # Create a Spark Session
        self.spark = SparkSession.builder.appName("MovieRecs").getOrCreate()

        # Parse Movie Links
        self.movieLinks = self.spark.read.text("movies/links.csv").rdd
        self.movieLinks = self.movieLinks.map(parseImdb)

        # Parse Movie Names and Categories
        self.movieAbouts = self.spark.read.text("movies/movies.csv").rdd
        self.movieAbouts = self.movieAbouts.map(parseAbouts)

        # Parse Movie Ratings
        self.movieRatings = self.spark.read.options(inferSchema=True, delimiter=',') \
            .csv("movies/ratings.csv").rdd


    def loadMovieImdbs(self, movieNames : dict):
        movieImdbsRDD = self.movieLinks.filter( lambda r: r[0] in movieNames.keys())
        movieImdbs = movieImdbsRDD.collectAsMap()
        return movieImdbs

    def loadMovieNames(self, Filter : list, Search : str):
        movieNamesRDD = self.movieAbouts.filter(lambda r: (r[1][0].lower().find(Search.lower()) != -1 ) &
                    (list(set(r[1][1].split(', ')).intersection(Filter)) != []) );

        movieNames = movieNamesRDD.collectAsMap();
        return movieNames;

    def GetBestMovies(self, movieNames : dict, Search : str):
        
        topMovieList = {}

        movieRatingsRDD = self.movieRatings.filter(lambda r: r[1] in movieNames.keys())        

        linesDF = movieRatingsRDD.toDF(["userID", "movieID", "rating", "timestamp"])
        averageRatings = linesDF.groupBy("movieID").avg("rating")

        if not Search:
            ratingCount = linesDF.groupBy("movieID").count().filter("count > 50");
            averageRatings = ratingCount.join(averageRatings,  ratingCount.movieID == averageRatings.movieID, "inner")

        averageRatings = averageRatings.orderBy("avg(rating)", ascending=False)
    
        for R in averageRatings.collect():
            topMovieList[R["movieID"]] = R["avg(rating)"]

        return topMovieList;


    @staticmethod
    def parseInput(line):
        fields = line.value.split(',')
        return Row(userID = int(fields[0]), movieID = int(fields[1]), rating = float(fields[2]))
    
    def GetUserRatings(self, userID):

        linesRDD = self.movieRatings.filter(lambda r: r[0] == userID)
        userRatings = {}
        for _, movieID, rating, _ in linesRDD.collect():
            userRatings[movieID] = rating;
        return userRatings

    def GetUserRecommendation(self, userid : int, movieNames : dict):
        # Get the raw data (RDD instead of Dataframe)
        lines = self.spark.read.text("movies/ratings.csv").rdd
        
        # Convert it to a RDD of Row Objects with (userID, movieID, rating)
        ratingsRDD = lines.map(self.parseInput)
        
        # filter it
        ratingsRDD = ratingsRDD.filter(lambda r: r[1] in movieNames.keys())
        
        # Convert to a Dataframe and then cache it
        # cache it because we use it more than once
        ratings = self.spark.createDataFrame(ratingsRDD).cache()

        # Create an ALS Collaborative filtering model from the complete dataset
        als = ALS(maxIter=5, regParam=0.01, userCol="userID", itemCol="movieID", ratingCol="rating")
        model = als.fit(ratings)

        # Print out ratings from user 0:
        # userRatings = ratings.filter(f"userID = {userid}")
        # for rating in userRatings.collect():
        #     print(movieNames[rating['movieID']], rating['rating'])

        # Find movies rated more than 100
        ratingsCounts = ratings.groupBy("movieID").count().filter("count > 50")

        # Construct a "test" dataframe for user 0 with every movie rated more than 100
        popularMovies = ratingsCounts.select("movieID").withColumn("userID", lit(userid))

        # Run our model on that list of popular movies
        recommendations = model.transform(popularMovies)

        # Get the top 20 movies with the highest predicted rating for this user
        topRecommendations = recommendations.sort(recommendations.prediction.desc())

        movieRecs = {}
        for recommendation in topRecommendations.collect():
            movieRecs[recommendation['movieID']] = recommendation['prediction']

        return movieRecs

app = Flask(__name__, template_folder="templates")
MM = MovieManager();
Genres = [
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "IMAX",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "Western",
    "War"
]
@app.route("/submitted", methods=["GET"])
def RatSubPage():
    ts = int(datetime.timestamp(datetime.now()))
    Rating = int(request.args.get("rating")) / 2
    movieID = int(request.args.get("movieID"))
    linesDF = MM.spark.createDataFrame(MM.movieRatings).cache()

    linesDF = linesDF.filter( (linesDF[0] != 0) | (linesDF[1] != movieID) );
    linesDF = linesDF.union(MM.spark.createDataFrame([(0, movieID, Rating, ts)], linesDF.columns))
    linesDF = linesDF.orderBy(linesDF[0].asc(), linesDF[1].asc())
    linesDF.coalesce(1).write.csv("movies/ratings2.csv", header=False, mode="overwrite", sep=",")
    os.system("hadoop fs -cat movies/ratings2.csv/part* > ratings.csv")
    os.system("hadoop fs -moveFromLocal -f ratings.csv movies/")
    os.system("hadoop fs -rm -r -f movies/ratings2.csv")

    MM.movieRatings = MM.spark.read.options(inferSchema=True, delimiter=',') \
        .csv("movies/ratings.csv").rdd

    return redirect("/")


@app.route('/rate', methods=['GET'])
def RatingPage():
    movieID = request.args.get('movieID')
    movieName = request.args.get('movieName')
    return render_template("stars.html", movieID=movieID, movieName=movieName)

@app.route('/', methods = ['GET', 'POST', 'DELETE'])
def MainPage():
    nMovies = 20;

    Filter = []
    Search = ""
    if request.method == 'POST':
        Search = request.form.get('movie-search')
        for Genre in request.values:
            if Genre in Genres:
                Filter.append(Genre);

    movieNames = MM.loadMovieNames(Genres if not Filter else Filter, Search);
    
    movieList = []

    if len(movieNames) > 0:
        userRatings = MM.GetUserRatings(0)
        movieRatings = MM.GetBestMovies(movieNames, Search);

        if userRatings and not Search:
            movieDic = MM.GetUserRecommendation(0, movieNames);
        else:
            movieDic = movieRatings

        movieImdbs = MM.loadMovieImdbs(movieNames);

        for movieID in movieDic.keys():
            userRating = "Unrated"

            if movieID not in movieNames.keys():
                continue;

            if movieID in userRatings.keys():
                userRating = userRatings[movieID]
        
            movieList.append((movieID, movieNames[movieID][0], movieNames[movieID][1], movieImdbs[movieID], userRating, str(movieRatings[movieID])[:4])) 
    
    nMovies = min(len(movieList), nMovies)

    return render_template("index.html", movieList=movieList, Genres=Genres, nMovies=nMovies);

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host="hadoop-master", port=port)