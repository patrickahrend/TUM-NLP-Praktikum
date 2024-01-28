import pickle


def load_pickle(file_path):
    """
    Load a pickle file.

    Parameters
    ----------
    file_path : str
        The path to the pickle file.

    Returns
    -------
    object
        The unpickled object.
    """
    with open(file_path, "rb") as file:
        return pickle.load(file)


def save_pickle(obj, file_path):
    """
    Save an object to a pickle file.

    Parameters
    ----------
    obj : object
        The object to be pickled.
    file_path : str
        The path where the pickle file will be saved.
    """
    with open(file_path, "wb") as file:
        pickle.dump(obj, file)
