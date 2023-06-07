import numpy as np
from PIL import Image


# copied from https://github.com/schaugf/HEnorm_python/blob/master/normalizeStaining.py

def normalizeStaining(img, Io=240, alpha=1, beta=0.15):
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

    """
    seems to be a reference matrix [v1, v2] where v1 and v2 are the reference
    color vectors. The matrix is used to produced the final image with the
    desired colors
    """
    # HERef = np.array([[0, 0.2159],
    #                  [0, 0.8012],
    #                  [1, 0.5581]])

    """ this seems to define the actual concentrations of the stain in the image. It is used
    to equalize the intensities of both channels to account for inequalities in staining.
    I have no idea about how that is for Hematoxylin and DAB. So set it to (1,1) or leave it out.
    It should not be relevant anyway, because we care about relative intensities in the DAB
    channel to find out about CYP expression."""

    #maxCRef = np.array([1, 1])

    # define height and width of image
    h, w, c = img.shape

    # reshape image
    img = img.reshape((-1, 3))

    # calculate optical density
    OD = -np.log((img.astype(float) + 1) / Io)

    # remove transparent pixels
    ODhat = OD[~np.any(OD < beta, axis=1)]

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

    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T

    # determine concentrations of the individual stains
    # This should be connected to the intensity by lambert beer or sth.
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]

    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    #tmp = np.divide(maxC, maxCRef) # normalization for ref concentrations, leave it out
    C2 = np.divide(C, maxC[:, np.newaxis])

    RC1 = np.reshape(C[0, :], (h, w, 1))  # intensities channel 1
    RC1N = np.reshape(C2[0, :], (h, w, 1))  # normalized intensities channel 1

    RC2 = np.reshape(C[1, :], (h, w, 1))  # intensities channel 2
    RC2N = np.reshape(C2[1, :], (h, w, 1))  # normalized intensities channel 2
    return RC1, RC1N, RC2, RC2N

    # recreate the image using reference mixing matrix
    # Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    # Inorm[Inorm >255] = 254
    # Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    # unmix hematoxylin and eosin
    H = np.multiply(Io, np.exp
    (np.expand_dims(-HERef[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))))
    H[H > 255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

    E = np.multiply(Io, np.exp
    (np.expand_dims(-HERef[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))))
    E[E > 255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)

    return Inorm, H, E
