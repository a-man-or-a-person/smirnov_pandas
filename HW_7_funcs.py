import math as m


def mse(answers: list, predictions: list) -> float:
    summ = sum([(ans - pred) ** 2 for ans, pred in zip(answers, predictions)])
    return (1 / min(len(answers), len(predictions))) * summ


def rmse(answers: list, predictions: list) -> float:
    return mse(answers, predictions) ** 0.5


def r2(answers: list, predictions: list) -> float:
    mean_y = sum(answers) / len(answers)
    summ1 = sum([(ans - pred) ** 2 for ans, pred in zip(answers, predictions)])
    summ2 = sum([(ans - mean_y) ** 2 for ans in answers])

    return 1 - (summ1 / summ2)


def mae(answers: list, predictions: list) -> float:
    summ = sum([abs(ans - pred) for ans, pred in zip(answers, predictions)])
    return (1 / min(len(answers), len(predictions))) * summ


def mape(answers: list, predictions: list) -> float:
    summ = sum([(abs(ans - pred) / abs(ans)) if ans != 0 else 0 for ans, pred in zip(answers, predictions)])
    return (1 / min(len(answers), len(predictions))) * summ


def smape(answers: list, predictions: list) -> float:
    summ = 0
    for ans, pred in zip(answers, predictions):
        summ += (2 * (ans - pred)) / (ans + pred)
    return (1 / min(len(answers), len(predictions))) * summ


def wape(answers: list, predictions: list) -> float:
    summ1 = sum([abs(ans - pred) for ans, pred in zip(answers, predictions)])
    return summ1 / sum([abs(x) for x in answers])


def rmsle(answers: list, predictions: list, c: float) -> float:
    summ = 0
    for ans, pred in zip(answers, predictions):
        summ += (m.log(ans + c) - m.log(pred + c)) ** 2
    return ((1 / min(len(answers), len(predictions))) * summ) ** 0.5
