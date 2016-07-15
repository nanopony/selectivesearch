from ._selectivesearch import _selectivesearch

def selectivesearch(image, threshold= 2000.0, min_size=200):
    """Performs Selective Search for regions with homogenuios colour/texture/etc

    Parameters
    ----------
    image : (M, N, 3) ndarray
        Colour image
    threshold : float, optional
    min_size: int, optional
        Minimum size of an region

    Returns
    -------
    H : List of tuples containing (ay,ax,by,bx) coordinates of an rectangle regions
    """
    
    return _selectivesearch(image, threshold= threshold, min_size = min_size)
