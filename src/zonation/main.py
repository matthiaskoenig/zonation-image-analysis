

if __name__ == "__main__":
    # read images

    # perform analysis

    # create figures/report (html)

    
    plot_image(data, cmap="gray")
    plot_image(data, cmap="tab20c")
    # plot_histogram(data)
    data_gauss = gauss_filter(data, sigma=10)
    plot_image(data_gauss, cmap="tab20c")
    # data_log = laplace_filter(data, ksize=100)
    # plot_image(data_log, cmap="tab20c")
    plot_histogram(data_gauss)
    # data_distance = watershed(data_gauss)
    # plot_image(data_distance)
