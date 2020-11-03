import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat, unumpy as unp
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy import constants as consts
from PIL import Image


def deviation(val, err, lit):
    return abs(lit - val) / err


def linear(x, a, b):
    return a * x + b


def lorentzian(x, a, x0, gamma):
    return a * (gamma ** 2 / ((x - x0) ** 2 + gamma ** 2)) / (np.pi * gamma)


def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def pseudo_voigt(x, eta, a1, x0, gamma, a2, mu, sigma):
    return eta * lorentzian(x, a1, x0, gamma) + (1 - eta) * gaussian(x, a2, mu, sigma)


def r_sq(data, fit):
    """
    R^2 Goodness of Fit Test
    Good fits return values close to 1.
    See also: https://en.wikipedia.org/wiki/Coefficient_of_determination
    """
    ss_res = np.sum((data - fit) ** 2)
    ss_tot = np.sum((data - np.mean(data)) ** 2)

    return 1 - (ss_res / ss_tot)


def test_r_sq():
    N = 300

    xnoise = np.random.uniform(0.0, 30.0, size=N)
    ynoise = np.random.uniform(0.0, 30.0, size=N)

    a = 3
    b = 5

    x = np.linspace(0.0, 5.0, N) + xnoise
    y = linear(x, a, b) + ynoise

    popt, pcov = curve_fit(linear, x, y)

    fit_a = ufloat(popt[0], np.sqrt(pcov[0][0]))
    fit_b = ufloat(popt[1], np.sqrt(pcov[1][1]))
    fit_y = linear(x, *popt)

    print("Fit Values:")
    print(f"a = {fit_a}")
    print(f"b = {fit_b}")
    print(f"r^2 = {r_sq(y, fit_y)}")

    plt.scatter(x, y, marker=".")
    plt.plot(x, fit_y, label="fit", color="orange")
    plt.plot(x, linear(x, a, b), label="analytical", color="red")
    plt.legend()
    plt.show()


def magnetic_field():
    current = np.array([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0])
    current_error = 0.3

    current = unp.uarray(current, current_error)

    b = np.array([
        [408, 404, 402],
        [450, 440, 446],
        [485, 492, 493],
        [524, 526, 531],
        [568, 566, 560],
        [581, 584, 588],
        [569, 567, 570],
        [530, 532, 534],
        [504, 499, 495],
        [461, 460, 457],
        [411, 416, 417],
    ])

    b_error = 0.02 * b

    b = unp.uarray(b, b_error)
    b = np.mean(b, axis=1)

    x = unp.nominal_values(current)
    y = unp.nominal_values(b)
    xerr = unp.std_devs(current)
    yerr = unp.std_devs(b)

    plt.figure(figsize=(10, 8))
    # rising current
    _x = x[:-5]
    _y = y[:-5]
    _xerr = xerr[:-5]
    _yerr = yerr[:-5]

    popt_rising, pcov_rising = curve_fit(linear, _x, _y, sigma=_yerr, absolute_sigma=True)

    r2 = r_sq(data=_y, fit=linear(_x, *popt_rising))

    print("r2:", r2)

    plt.errorbar(x=_x, y=_y, xerr=_xerr, yerr=_yerr, label="rising", ls="None", color="limegreen")
    fit_x = np.linspace(min(x), max(x), 1000)
    plt.plot(fit_x, linear(fit_x, *popt_rising), label="rising fit", color="green")

    # falling current
    _x = x[-5:]
    _y = y[-5:]
    _xerr = xerr[-5:]
    _yerr = yerr[-5:]

    popt_falling, pcov_falling = curve_fit(linear, _x, _y, sigma=_yerr, absolute_sigma=True)

    r2 = r_sq(data=_y, fit=linear(_x, *popt_falling))

    print("r2:", r2)

    plt.errorbar(x=_x, y=_y, xerr=_xerr, yerr=_yerr, label="falling", ls="None", color="orange")
    plt.plot(fit_x, linear(fit_x, *popt_falling), label="falling fit", color="red")

    plt.title("Hysteresis Effect in the Magnetic Field")
    plt.xlabel("I / A")
    plt.ylabel("B / mT")
    plt.legend()
    plt.savefig("figures/hysteresis.png")
    plt.show()

    return popt_falling, pcov_falling


def estimate_fwhm(x, y, yp):
    # data must be without background
    x = [_x for i, _x in enumerate(x) if y[i] > yp / 2]
    return x[-1] - x[0]


def autofit(fit_func, x, y, xp, yp):
    """ Automatically find starting values for the given peak and fit """

    a = (y[-1] - y[0]) / (x[-1] - x[0])
    b = y[0] - a * x[0]
    eta = 0.5
    a1 = a2 = yp - linear(xp, a, b)
    x0 = xp
    mu = xp
    gamma = estimate_fwhm(x, y - linear(x, a, b), yp - linear(xp, a, b)) / 2
    sigma = 0.8493 * gamma

    p0 = [a, b, eta, a1, x0, gamma, a2, mu, sigma]

    # print("p0 =", p0)

    # in addition to the mathematical bounds, we add bounds for x0 and mu which should lie very close to xp
    # since otherwise curve_fit will try to send one to infinity if the respective distribution has
    # low influence on the peak which would be unphysical
    lower_bounds = [-np.inf, -np.inf, 0.0, 0.0, xp - 30, 0.0, 0.0, xp - 30, 0.0]
    upper_bounds = [np.inf, np.inf, 1.0, np.inf, xp + 30, np.inf, np.inf, xp + 30, np.inf]
    bounds = lower_bounds, upper_bounds

    popt, pcov = curve_fit(fit_func, x, y, p0=p0, bounds=bounds, maxfev=10000)

    return popt, pcov


def fit_bounds(i, xp, xtroughs):
    """ Automatically find fitting bounds for the given peak """
    def _fit_min():
        result = None
        for element in xtroughs:
            if element < xp:
                result = element
            else:
                return result

    def _fit_max():
        result = None
        for element in reversed(xtroughs):
            if element > xp:
                result = element
            else:
                return result

    fit_min = _fit_min()
    fit_max = _fit_max()

    which = i % 3
    if which == 0:
        # right peak
        offset = (fit_max - fit_min) // 3
        return fit_min, fit_max - offset
    elif which == 1:
        # left peak
        offset = (fit_max - fit_min) // 3
        return fit_min + offset, fit_max
    else:
        # middle peak
        return fit_min, fit_max


def sigma_pi(filename, plot=False):
    """
    Autofit all Sigma and Pi Peaks in the given file
    :return: Fit Values for each peak
    """
    x, y = np.loadtxt(filename, skiprows=1, unpack=True)

    # remove nan values
    nans = np.isnan(y)
    x = x.astype(int)[~nans]
    y = y[~nans]

    # restrict the minimum height of peaks to not match valleys
    # and set a peak width of 5 samples to avoid multiple matches or matches as a result from slight noise
    xpeaks = find_peaks(y, height=-1.35e7, width=5)[0]
    ypeaks = y[xpeaks]

    # have to manually add the final value of the dataset as a trough
    xtroughs = np.append(find_peaks(-y, width=5)[0], x[-1])
    ytroughs = y[xtroughs]

    if plot:
        plt.figure(figsize=(10, 8))
        plt.plot(x, y, label="data")
        plt.scatter(xtroughs, ytroughs, color="orange", marker="x")
        plt.show()

    def fit_func(x, a, b, *args, **kwargs):
        return a * x + b + pseudo_voigt(x, *args, **kwargs)

    for i, (xp, yp) in enumerate(zip(xpeaks, ypeaks)):
        fit_min, fit_max = fit_bounds(i, xp, xtroughs)
        fit_x = x[fit_min:fit_max]
        fit_y = y[fit_min:fit_max]
        popt, pcov = autofit(fit_func, fit_x, fit_y, xp, yp)

        r2 = r_sq(fit_y, fit_func(fit_x, *popt))

        yield popt, pcov, r2

        if plot:
            fit_color = "green" if r2 > 0.995 else "orange" if r2 > 0.99 else "red"
            dense_fit_x = np.linspace(fit_min, fit_max, 100)
            plt.figure(figsize=(16, 12))
            plt.plot(x, y, label="data")
            plt.scatter([fit_min, fit_max], [y[fit_min], y[fit_max]], color="orange", marker="x")
            plt.plot(dense_fit_x, fit_func(dense_fit_x, *popt), label="fit", color=fit_color)
            plt.xlabel("x / px")
            plt.ylabel("Intensity / counts")
            plt.title(filename)
            plt.show()


def part_1(cd_wavelength, popt_b, pcov_b):
    # magnetic_field()

    for current in range(8, 13 + 1):
        print(f"=== CURRENT: {current} A ===")
        filename = "data/" + str(current) + " A.xls"

        # acquire fit data
        data = sigma_pi(filename, plot=False)

        # filter out bad fits
        data = filter(lambda d: d[2] > 0.99, data)

        # collect data
        data = list(data)

        # split into sigma and pi
        pi_data = data[2::3]
        sigma_data = data[1::3]

        # acquire orders of interference
        k0 = 1e5
        kn = k0 + len(pi_data)
        ks = np.linspace(k0, kn, len(pi_data))

        # plot orders of interference against position of pi-lines
        pi_xs = np.array([pi[0][4] for pi in pi_data])
        sig_xs = np.array([sig[0][4] for sig in sigma_data])

        plt.figure(figsize=(10, 8))
        plt.scatter(ks, pi_xs, label="pi", marker="x")
        plt.scatter(ks, sig_xs, label="sigma", marker="x")

        # polynomial order, we found n = 2 to be sufficient (r^2 > 0.999)
        n_pi = 2
        n_sig = 2

        popt_pi = np.polyfit(ks, pi_xs, n_pi)
        popt_sig = np.polyfit(ks, sig_xs, n_sig)

        fit_x = np.linspace(ks[0], ks[-1], 1000)

        print("[pi]: r^2 =", r_sq(pi_xs, np.poly1d(popt_pi)(ks)))
        print("[sig]: r^2 =", r_sq(sig_xs, np.poly1d(popt_sig)(ks)))

        plt.plot(fit_x, np.poly1d(popt_pi)(fit_x), label="pi fit")
        plt.plot(fit_x, np.poly1d(popt_sig)(fit_x), label="sigma fit")

        Delta_a = np.poly1d(popt_pi)(1)
        delta_a = np.abs(np.poly1d(popt_sig)(1) - Delta_a)

        print("Delta_a =", Delta_a)
        print("delta_a =", delta_a)

        d = 4.04e6  # nm
        n = 1.4567
        Delta_lambda = cd_wavelength ** 2 / (2 * d * np.sqrt(n ** 2 - 1))  # m
        delta_lambda = delta_a / Delta_a * Delta_lambda  # m

        print(f"Delta_lambda = {Delta_lambda * 1e3} pm")
        print(f"delta_lambda = {delta_lambda * 1e3} pm")

        # TODO
        Delta_E = consts.h * consts.c * (1 / (cd_wavelength * 1e-9) - 1 / (cd_wavelength * 1e-9 + delta_lambda))

        errors = [pcov_b[i][i] for i in range(len(popt_b))]
        B = linear(current, *unp.uarray(popt_b, errors)) * 1e-3  # T

        print("B =", B)

        mu_B = Delta_E / B

        print("mu_B =", mu_B)

        plt.title("Sigma and Pi Peak Positions")
        plt.xlabel("Order of Interference")
        plt.ylabel("position / px")
        plt.legend()
        plt.savefig(f"figures/sigma_pi_{current}A.png")
        plt.show()


def avg_img(name, n0=1, n=5, ft="jpg"):
    fp = "data/"

    return np.average(np.array([
        np.asarray(Image.open(f"{fp}{name}{i}.{ft}").rotate(1.2))
        for i in range(n0, n)
    ]), axis=0)


def vintegrate_avg_img(name, bg_name, *args, **kwargs):
    a = avg_img(name, *args, **kwargs) - avg_img(bg_name, *args, **kwargs)
    return np.sum(a, axis=0).take(0, axis=1)


def part_2():

    integrated_a = vintegrate_avg_img("a", "b")
    integrated_cd = vintegrate_avg_img("cd a", "cd b")
    x = np.arange(len(integrated_a))

    p0s = [
        # a, b, eta, a1, x0, gamma, a2, mu, sigma
        [0, 0, 0.5, 3000, 10, 5, 3000, 10, 4],
        [0, 0, 0.5, 10000, 230, 5, 10000, 230, 4],
        [0, 0, 0.5, 37750, 1104, 5, 37750, 1104, 4],
        [5000 / 30, -21000, 0.5, 20000, 1265, 5, 20000, 1265, 4]
    ]

    fit_lims = [
        (0, 23),
        (218, 241),
        (1091, 1115),
        (1251, 1277)
    ]

    plt.figure(figsize=(10, 8))
    plt.plot(x, integrated_a, label="Neon")
    # plt.plot(x, integrated_cd, label="Cadmium")

    def fit_func(_x, _a, _b, *args, **kwargs):
        return _a * _x + _b + pseudo_voigt(_x, *args, **kwargs)

    positions = []

    for p0, (fit_min, fit_max) in zip(p0s, fit_lims):
        fit_x = x[fit_min:fit_max]
        fit_y = integrated_a[fit_min:fit_max]

        # in addition to the mathematical bounds, we add bounds for x0 and mu which should lie very close to xp
        # since otherwise curve_fit will try to send one to infinity if the respective distribution has
        # low influence on the peak which would be unphysical
        xp = p0[4]
        lower_bounds = [-np.inf, -np.inf, 0.0, 0.0, xp - 5, 0.0, 0.0, xp - 5, 0.0]
        upper_bounds = [np.inf, np.inf, 1.0, np.inf, xp + 5, np.inf, np.inf, xp + 5, np.inf]
        bounds = lower_bounds, upper_bounds

        popt, pcov = curve_fit(fit_func, fit_x, fit_y, p0=p0, bounds=bounds, maxfev=10000)

        eta = ufloat(popt[2], pcov[2][2])
        x0 = ufloat(popt[4], pcov[4][4])
        mu = ufloat(popt[7], pcov[7][7])
        positions.append(eta * x0 + (1 - eta) * mu)

        r2 = r_sq(fit_y, fit_func(fit_x, *popt))

        print(r2)

        fit_color = "green" if r2 > 0.995 else "orange" if r2 > 0.99 else "red"

        plot_x = np.linspace(fit_min, fit_max, 1000)
        plt.plot(plot_x, fit_func(plot_x, *popt), color=fit_color)

    plt.title("Neon Line Spectrum")
    plt.xlabel("Position / px")
    plt.ylabel("Intensity / counts")
    # plt.legend()
    plt.savefig("figures/ne_spectrum.png")
    plt.show()
    # plt.close()

    # camera was upside down during measurement, so the left side is higher wavelengths
    lambda_neon = [653.28822, 650.65281, 640.2248, 638.29917]

    popt_calibration, pcov_calibration = curve_fit(linear, unp.nominal_values(positions), lambda_neon)
    l = linear(x, *popt_calibration)

    plt.figure(figsize=(10, 8))
    plt.plot(unp.nominal_values(positions), lambda_neon, lw=0, marker="x", label="Neon Line Data")

    temp_plot_x = np.linspace(0, 1300, 10000)
    plt.plot(temp_plot_x, linear(temp_plot_x, *popt_calibration), label="fit")
    del temp_plot_x

    plt.title("Calibration")
    plt.xlabel("position / px")
    plt.ylabel(r"$\lambda$ / nm")
    plt.legend()
    plt.savefig("figures/calibration.png")
    plt.show()

    # a, b, eta, a1, x0, gamma, a2, mu, sigma
    p0 = [0, 0, 0.5, 36500, 802.5, 6, 36500, 802.5, 5]
    fit_min = 750
    fit_max = 850

    fit_x = x[fit_min:fit_max]
    fit_y = integrated_cd[fit_min:fit_max]

    xp = p0[4]
    lower_bounds = [-np.inf, -np.inf, 0.0, 0.0, xp - 5, 0.0, 0.0, xp - 5, 0.0]
    upper_bounds = [np.inf, np.inf, 1.0, np.inf, xp + 5, np.inf, np.inf, xp + 5, np.inf]
    bounds = lower_bounds, upper_bounds

    popt, pcov = curve_fit(fit_func, fit_x, fit_y, p0=p0, bounds=bounds, maxfev=10000)

    r2 = r_sq(fit_y, fit_func(fit_x, *popt))

    print("r2 =", r2)

    eta = ufloat(popt[2], pcov[2][2])
    x0 = ufloat(popt[4], pcov[4][4])
    mu = ufloat(popt[7], pcov[7][7])
    cd_position = eta * x0 + (1 - eta) * mu

    cd_wavelength = linear(cd_position, *popt_calibration)

    cd_wavelength_lit = 643.8470  # nm

    print("cd position =", cd_position)
    print("cd wavelength =", cd_wavelength)
    print("deviation:", deviation(cd_wavelength.nominal_value, cd_wavelength.std_dev, cd_wavelength_lit), "sigma")

    plt.figure(figsize=(10, 8))
    plt.plot(l, integrated_cd, label="data")

    plot_x = np.linspace(fit_min, fit_max, 10000)
    plt.plot(linear(plot_x, *popt_calibration), fit_func(plot_x, *popt), label="fit")

    plt.title("Cadmium Spectrum")
    plt.xlabel(r"$\lambda$ / nm")
    plt.ylabel("Intensity / counts")
    plt.xlim(643.2, 644.6)
    plt.legend()
    plt.savefig("figures/cd_spectrum.png")
    plt.show()

    return cd_wavelength


def main(argv: list) -> int:
    popt, pcov = magnetic_field()
    cd = part_2()
    part_1(cd, popt, pcov)
    return 0


if __name__ == "__main__":
    import sys

    main(sys.argv)
