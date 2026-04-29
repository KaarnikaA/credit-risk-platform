def probability_to_score(prob):
    """
    convert model probablity to credit scors(FICO)
    """
    score = 850 - (prob * 550)
    return int(score)