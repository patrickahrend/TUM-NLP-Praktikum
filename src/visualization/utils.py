import pickle


def load_pickle(file_path: str) -> object:
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


def save_pickle(obj: object, file_path: str) -> None:
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
