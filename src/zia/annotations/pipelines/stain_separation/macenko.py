import numpy as np

from zia.annotations.workflow_visualizations.util.image_plotting import plot_pic


# copied from https://github.com/schaugf/HEnorm_python/blob/master/normalizeStaining.py


def normalizeStaining(img, gs_threshold: int, Io=240, alpha=1):
    ''' Normalize staining appearence of H&E stained images

    Example use:
        see test.py

    Input:
        I: RGB input image
        Io: (optional) transmitted light intensity

    Output:
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image

    Reference:
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    '''

    HERef = np.array([[1, 0],
                      [1, 1],
                      [0, 1]])

    # maxCRef = np.array([1, 1])

    # define height and width of image
    h, w, c = img.shape

    # reshape image
    img = img.reshape((-1, 3))

    # calculate optical density
    OD = -np.log((img.astype(float) + 1) / Io)

    gsv = [0.587, 0.114, 0.299]
    gs_img = img.dot(gsv)
    # print(gs_img)
    ODhat = OD[gs_img < gs_threshold]
    # remove transparent pixels
    # ODhat = OD[~np.any(OD < beta, axis=1)]

    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

    # eigvecs *= -1

    # project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    That = ODhat.dot(eigvecs[:, 1:3])

    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    print(HE)
    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]

    # print(C.shape)
    # print(np.min(C[0, :]), np.min(C[1, :]))

    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])

    # we do not need those reference max concentrations. We don't know anyway
    # tmp = np.divide(maxC, maxCRef)
    # C2 = np.divide(C, tmp[:, np.newaxis])

    # That should actually contain the information about the cyps (concentration)
    # as given by Lambert Beer log(I0/I) = e*c*d where c is concentration
    # However, some pixels have negative concentrations
    C2 = np.divide(C, maxC[:, np.newaxis])

    print(C2.shape)

    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    # unmix hematoxylin and eosin
    # CHANGED: Instead of using reference matrix for mixing, the image is just
    # the concentration is just exponentiated into a single channel
    #
    H = np.multiply(Io, np.exp(-C2[0, :]))
    H[H > 255] = 254
    H = np.reshape(H.T, (h, w, 1)).astype(np.uint8)

    E = np.multiply(Io, np.exp(-C2[1, :]))
    E[E > 255] = 254
    E = np.reshape(E.T, (h, w, 1)).astype(np.uint8)

    return Inorm, H, E


def calculate_stain_matrix(pxi: np.ndarray, Io=240, alpha=1) -> np.ndarray:
    """
    calculates the stain base vectors
    @param pxi: pixels of interest as np array of RGB tuples with shape(h*w, 3)
    @param Io: transmitted light intensity
    @param alpha: percentile to find robust maxima
    """

    od = -np.log((pxi.astype(float) + 1) / Io)

    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(od.T))

    # eigvecs *= -1

    # project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    t_hat = od.dot(eigvecs[:, 1:3])

    phi = np.arctan2(t_hat[:, 1], t_hat[:, 0])

    min_phi = np.percentile(phi, alpha)
    max_phi = np.percentile(phi, 100 - alpha)

    v_min = eigvecs[:, 1:3].dot(np.array([(np.cos(min_phi), np.sin(min_phi))]).T)
    v_max = eigvecs[:, 1:3].dot(np.array([(np.cos(max_phi), np.sin(max_phi))]).T)

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if v_min[0] > v_max[0]:
        stain_matrix = np.array((v_min[:, 0], v_max[:, 0])).T
    else:
        stain_matrix = np.array((v_max[:, 0], v_min[:, 0])).T

    return stain_matrix


def deconvolve_image(pxi, stain_matrix: np.ndarray, Io=240, maxC = None):
    """
    deconvolution of the stains for the pixels of interest.
    @param pxi: pixels of interest
    @param stain_matrix: m x 2 matrix where m is rgb channel for 2 stain colors
    @param Io: transmission intensity
    @return: Io, he, dab np arrays of shapes (m*n, 3), (m*n, 1), (m*n, 1)
    """

    # rows correspond to channels (RGB), columns to OD values
    y = -np.log((pxi.astype(float) + 1) / Io).T

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(stain_matrix, y, rcond=None)[0]

    # print(C.shape)
    # print(np.min(C[0, :]), np.min(C[1, :]))

    # normalize stain concentrations
    if maxC is None:
        maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])

    # we do not need those reference max concentrations. We don't know anyway
    # tmp = np.divide(maxC, maxCRef)
    # C2 = np.divide(C, tmp[:, np.newaxis])

    # That should actually contain the information about the cyps (concentration)
    # as given by Lambert Beer log(I0/I) = e*c*d where c is concentration
    # However, some pixels have negative concentrations
    C2 = np.divide(C, maxC[:, np.newaxis])

    # recreate the image using reference mixing matrix
    Inorm = reconstruct_pixels(C2)

    # unmix hematoxylin and eosin
    # CHANGED: Instead of using reference matrix for mixing, the image is just
    # the concentration is just exponentiated into a single channel
    #
    H = create_single_channel_pixels(C2[0, :])

    E = create_single_channel_pixels(C2[1, :])

    return [Inorm, H, E], maxC


def create_single_channel_pixels(concentrations: np.ndarray, Io=240):
    """
    reconstructs the pixel values for one stain using the concentrations from
    the deconvolution
    @param concentrations: shape (m, ) matrix containing the concentration of the stain
    @param Io: transmission intensity
    @return:
    """
    i = np.multiply(Io, np.exp(-concentrations)).astype(np.uint8).T.reshape(-1, 1)
    i[i > 255] = 254
    return i


def reconstruct_pixels(concentrations: np.ndarray, refrence_matrix=None, Io=240):
    """
    reconstructs the image pixels using the reference_matrix and the concentrations
    from the deconvolution
    @param concentrations: shape (m, 2) matrix containg the concentrations of the stains
    @param refrence_matrix: shape (3,2) matrix containing RGB color vectors for the stains
    @param Io: transmission intensity
    @return:
    """
    HERef = np.array([[0, 1],
                      [1, 1],
                      [1, 0]])

    if refrence_matrix is not None:
        HERef = refrence_matrix

    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(concentrations)))
    Inorm[Inorm > 255] = 254
    Inorm = Inorm.astype(np.uint8).T.reshape(-1, 3)
    return Inorm
