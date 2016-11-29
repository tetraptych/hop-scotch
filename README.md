# hop/scotch

hop/scotch is a personalized whiskey recommendation engine developed using the ratings at [WhiskyBase](http://www.whiskybase.com). The foundation of the service is an instance of GraphLab's [Ranking Factorization Recommender](https://turi.com/products/create/docs/generated/graphlab.recommender.ranking_factorization_recommender.RankingFactorizationRecommender.html#graphlab.recommender.ranking_factorization_recommender.RankingFactorizationRecommender).

The model has two distinct modalities.
* When sufficient user preference information is available, the app uses this input and the pre-computed matrix factors to make predictions via straightforward collaborative filtering techniques.  
* In the absence of such data, the app takes a different approach. For each possible item, it computes two similarity scores: one based on user-item interactions and one based on topic modeling via Latent Dirichlet Allocation. These scores are aggregated dynamically according to the number of ratings provided and ranked to provide recommendations.
