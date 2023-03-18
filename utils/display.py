import numpy as np


def multivariate_normal(x, d, mean, covariance):
    """pdf of the multivariate normal distribution."""
    x_m = x - mean
    return (
        1.0
        / (np.sqrt((2 * np.pi) ** d * np.linalg.det(covariance)))
        * np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2)
    )


# Plot bivariate distribution
def generate_surface(mean, covariance, d):
    """Helper function to generate density surface."""
    nb_of_x = 1000  # grid size
    x1s = np.linspace(-50, 50, num=nb_of_x)
    x2s = np.linspace(-50, 50, num=nb_of_x)
    x1, x2 = np.meshgrid(x1s, x2s)  # Generate grid
    pdf = np.zeros((nb_of_x, nb_of_x))
    # Fill the cost matrix for each combination of weights
    for i in range(nb_of_x):
        for j in range(nb_of_x):
            pdf[i, j] = multivariate_normal(
                np.array([x1[i, j], x2[i, j]]), d, mean, covariance
            )
    return x1, x2, pdf  # x1, x2, pdf(x1,x2)
