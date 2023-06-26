import numpy as np


# copied from https://github.com/schaugf/HEnorm_python/blob/master/normalizeStaining.py


def normalize_staining(image, Io=240, alpha=1, beta=0.15):
    # define height and width of image
    optical_density = calculate_optical_density(image, Io)
    stain_matrix = calculate_stain_matrix(optical_density, alpha, beta)

    return deconvolve_image(optical_density, image.shape, stain_matrix)


def deconvolve_image(optical_density: np.ndarray, image_shape: tuple[int, int, int], stain_matrix: np.ndarray) -> \
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    uses the stain base vectors to deconvolve the image
    returns tuple of (hematoxylin, hematoxylin_normalized, dab, dab_normalized)
    """
    h, w, c = image_shape

    y = np.reshape(optical_density, (-1, 3)).T

    # determine concentrations of the individual stains
    # This should be connected to the intensity by lambert beer or sth.
    concentrations = np.linalg.lstsq(stain_matrix, y, rcond=None)[0]

    # normalize stain concentrations
    max_concentration = np.array([np.percentile(concentrations[0, :], 99), np.percentile(concentrations[1, :], 99)])
    # tmp = np.divide(max_concentration, maxCRef) # normalization for ref concentrations, leave it out
    normalized_concentrations = np.divide(concentrations, max_concentration[:, np.newaxis])

    hematoxylin = np.reshape(concentrations[0, :], (h, w, 1))  # intensities channel 1
    hematoxylin_norm = np.reshape(normalized_concentrations[0, :], (h, w, 1))  # normalized intensities channel 1

    dab = np.reshape(concentrations[1, :], (h, w, 1))  # intensities channel 2
    dab_norm = np.reshape(normalized_concentrations[1, :], (h, w, 1))  # normalized intensities channel 2
    return hematoxylin, hematoxylin_norm, dab, dab_norm


def calculate_stain_matrix(od: np.ndarray, alpha=1, beta=0.15) -> np.ndarray:
    """Normalize staining appearence of H&E stained images

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
        """

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

    # remove transparent pixels
    od_hat = od[~np.any(od < beta, axis=1)]

    # compute eigenvectors
    eig_vals, eig_vecs = np.linalg.eigh(np.cov(od_hat.T))

    # eig_vecs *= -1

    # project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    t_hat = od_hat.dot(eig_vecs[:, 1:3])

    phi = np.arctan2(t_hat[:, 1], t_hat[:, 0])

    min_phi = np.percentile(phi, alpha)
    max_phi = np.percentile(phi, 100 - alpha)

    v_min = eig_vecs[:, 1:3].dot(np.array([(np.cos(min_phi), np.sin(min_phi))]).T)
    v_max = eig_vecs[:, 1:3].dot(np.array([(np.cos(max_phi), np.sin(max_phi))]).T)

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second -> in this case dab
    if v_min[0] > v_max[0]:
        stain_vectors = np.array((v_min[:, 0], v_max[:, 0])).T
    else:
        stain_vectors = np.array((v_max[:, 0], v_min[:, 0])).T

    return stain_vectors


def calculate_optical_density(image: np.ndarray, transmission_intensity: float = 240) -> np.ndarray:
    """
    calculates the optical density over an image array
    img is of shape (m, n, 3)
    the return array is of shape (m * n, 3)
    """
    # reshape image
    image = image.reshape((-1, 3))

    # calculate optical density
    return -np.log((image.astype(float) + 1) / transmission_intensity)
