This problem set examines two models, latent factor analysis and hidden Markov models.  The original homework assignment from the course provided implementations of both models; we implemented them and the EM algorithm for their training from scratch in Python.  The latent factor analysis is used to predict movie ratings of users using the MovieLens database; it is compared to a baseline PCA using a root-mean-square metric.

Hidden Markov models are trained on text snippets from Alice's Adventures in Wonderland, and used for predicting missing letters from a test dataset; the results are not very compelling - there seems to be either insufficient training data or training time for such a high-dimensional model.

The models are implemented in hw9.py and the pset questions are answered in pset9.ipynb
