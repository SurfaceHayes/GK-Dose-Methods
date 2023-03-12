import numpy as np
from tensorflow.python.keras import backend as K

def data_split(val_split, test_split, data_len):
    """
    Function
    ----------
    Calcuate the ratio of fitting to validation from a fraction

    Parameters
    ----------
    val_split: float
        A float representing the split of fitting:validation
    data_len: int
        Represents the total length of the data set

    Returns
    ----------
    fit_num: int
        An integer for the length of the fitting data set
    val_num: int
        An integer for the length of the validation data set
    """
    val_num = round(val_split*data_len)
    test_num = round(test_split*data_len)
    fit_num = data_len - val_num - test_num
    return(fit_num, val_num, test_num)


def print_time(tot_time):
    """
    Function
    ----------
    Prints the elapsed time in the format of hours:minutes:seconds

    Parameters
    ----------
    tot_time: float
        The total elapsed time that should be printed
    """

    m, s = divmod(tot_time, 60)
    h, m = divmod(m, 60)
    print("THE ELAPSED TIME IS: %d:%02d:%02d" % (h, m, s))

def dice_coef(y_true, y_pred, smooth=0):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    @url: https://gist.github.com/wassname/7793e2058c5c9dacb5212c0ac0b18a8a
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def jaccard_index(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return(intersection / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred),-1) - intersection))

def jaccard_loss(y_true, y_pred):
    return(1-jaccard_index(y_true, y_pred))

def jaccard_distance(y_true, y_pred, smooth=1):
    """
    ref. https://stackoverflow.com/questions/49284455/keras-custom-function-implementing-jaccard
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

