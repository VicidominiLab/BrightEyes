# Origin: VicidominiLab/BrightEyes-FFS, src/brighteyes_ffs/tools/cast_data.py at f6d640d.
def cast_data(data, type='int64'):
    """
    Convert data array to type cast

    Parameters
    ----------
    data : np.array
        Data array.
    type : string, optional
        data type. The default is 'int64'.

    Returns
    -------
    data : np.array
        Same data array, but typecasted.

    """
    
    data = data.astype(type)
    return data
