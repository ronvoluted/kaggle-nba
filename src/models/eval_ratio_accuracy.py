import random
import statistics
from sklearn.metrics import accuracy_score

def ratioAccuracy(y_data, loops=10):
    """Evaluate Model:
        Return a baseline accuracy score based on ratio of classifications

        Parameters
        ----------
        y_data : list / array
            Data to generate ratio prediction from
        k : int
            Number of loops to perform for averaging score

        Returns
        -------
        accuracy_score
            Accuracy score of predicting based on ratio

        Usage
        -----
        ratioAccuracy(y_train)
    """

    preds = []
    proba = y_data.value_counts()[1] / len(y_data)
    scores = []

    for _ in range(loops):
        for _ in range(len(y_data)):
            preds.append(1 if random.random() < proba else 0)
        scores.append(accuracy_score(y_data, preds))
        preds = []
        
    return statistics.mean(scores)