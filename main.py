import seaborn
import numpy as np
import matplotlib.pyplot as plt
from numba import jit


def test(t):
    gamma = 1.4
    n = 360
    dt = 5 * 10**-5  # less than 10**-4
    n_ite = int(t / dt)

    rl = 1.
    pl = 1.
    vl = 0.

    rr = 0.125
    pr = 0.1
    vr = 0.

    ratio = rl / rr
    nr = int(n / (ratio + 1))
    nl = int(n - nr)

    x_min = -0.5
    x_max = 0.5

    zr = np.fliplr([np.linspace(x_max, 0., nr)])[0]
    zl = np.linspace(x_min, 0., nl, endpoint=False)

    dxr = (zr[2] - zr[1])
    dxl = (zl[2] - zl[1])
    h = 2 * dxr

    #DO THIS
    @jit
    def kernel_cubic(xi, xj):
        q = np.abs(xi - xj) / h
        if q <= 1. or q == 0.:
            W = 2. / (3 * h) * (1. - 3 / 2 * q**2 * (1 - q / 2))
        elif 1. < q < 2.:
            W = 2. / (12 * h) * (2 - q)**3
        else:
            W = 0.
        return W

    #DO THIS
    @jit
    def diff_W(xi, xj):
        q = np.abs(xi - xj) / h

        if q <= 1.:
            dW = (3 * q**2 - 4 * q) / (2 * h)
        elif 1. < q < 2.:
            dW = - (2 - q)**2 / (2 * h)
        else:
            dW = 0.

        if xi > xj:
            der = 1 / h
        elif xi < xj:
            der = -1 / h
        else:
            der = 0.

        return dW * der

    alpha = 1.
    beta = 1.
    eps = 0.1
    neta = eps * h

    @jit
    def pi(r, v, z, p, i, j):

        x = (z[i] - z[j])

        if x * (v[i] - v[j]) <= 0:
            ca = ((np.abs(gamma * p[i] / r[i]))**0.5 +
                  (np.abs(gamma * p[j] / r[j]))**0.5) / 2
            ra = (r[i] + r[j]) / 2
            mu = h * (v[i] - v[j]) * x / (np.abs(x)**2 + neta**2)
            pia = (-alpha * ca * mu + beta * mu**2) / ra
        else:
            pia = 0.

        return pia

    @jit
    def sumden(z, m):
        nc = len(z)
        dr = np.zeros(nc)
        for i in range(nc):
            for j in range(nc):
                dr[i] += m[j] * kernel_cubic(z[i], z[j])
        return dr

    @jit
    def rhs(r, v, z, p, m):
        nc = len(z)
        dv = np.zeros(nc)
        de = np.zeros(nc)
        for i in range(nc):
            if -.35 < z[i] < .35:
                for j in range(nc):
                    av = pi(r, v, z, p, i, j)
                    dW = diff_W(z[i], z[j])

                    calc = (p[j] / r[j]**2 + p[i] / r[i]**2 + av)
                    calc1 = (p[i] / r[i]**2 + av)

                    dv[i] += - m[j] * calc * dW
                    de[i] += m[j] / 2 * calc1 * (v[i] - v[j]) * dW
            else:
                dv[i] = 0.
                de[i] = 0.

        return dv, de

    #DO THIS
    @jit
    def vel(r, u, z, e_con, m):
        v = np.zeros_like(u)
        for i in range(len(u)):
            v[i] = u[i]
            for j in range(len(z)):
                v[i] += e_con * (m[j] * 2 / (r[j] + r[i])) * (u[j] - u[i]
                                                              ) * kernel_cubic(z[i], z[j])
        return v

    @jit
    def mirror(r, v, e, z, p, m):
        k = 6.
        mask1 = (z > x_min) & (z < x_min + k * h)
        mask2 = (z > x_max - k * h) & (z < x_max)

        z1 = np.fliplr([z[mask1] + 2 * (x_min - z[mask1])])[0]
        r1 = np.fliplr([r[mask1]])[0]
        v1 = 1 * np.fliplr([v[mask1]])[0]
        e1 = np.fliplr([e[mask1]])[0]
        p1 = np.fliplr([p[mask1]])[0]
        m1 = np.fliplr([m[mask1]])[0]

        z2 = np.fliplr([z[mask2] + 2 * (x_max - z[mask2])])[0]
        r2 = np.fliplr([r[mask2]])[0]
        v2 = 1 * np.fliplr([v[mask2]])[0]
        e2 = np.fliplr([e[mask2]])[0]
        p2 = np.fliplr([p[mask2]])[0]
        m2 = np.fliplr([m[mask2]])[0]

        z = np.append(z1, z)
        z = np.append(z, z2)

        r = np.append(r1, r)
        r = np.append(r, r2)

        v = np.append(v1, v)
        v = np.append(v, v2)

        e = np.append(e1, e)
        e = np.append(e, e2)

        p = np.append(p1, p)
        p = np.append(p, p2)

        m = np.append(m1, m)
        m = np.append(m, m2)

        return r, v, e, z, p, m

    @jit
    def scrap(r, v, e, z, p, m):
        mask = (z > x_min) & (z < x_max)
        z = z[mask]
        r = r[mask]
        v = v[mask]
        e = e[mask]
        p = p[mask]
        m = m[mask]

        return r, v, e, z, p, m

    @jit
    def update(r, v, e, z, p, m):
        r, v, e, z, p, m = mirror(r, v, e, z, p, m)

        r = sumden(z, m)
        p = (gamma - 1) * r * e

        dv, de = rhs(r, v, z, p, m)

        v = v + dv * dt
        e = e + de * dt

        v_mov = vel(r, v, z, 0.5, m)
        z = z + v_mov * dt

        r, v, e, z, p, m = scrap(r, v, e, z, p, m)
        return r, v, e, z, p, m

    z = np.append(zl, zr)
    p = np.append(pl * np.ones_like(zl), pr * np.ones_like(zr))
    r = np.append(rl * np.ones_like(zl), rr * np.ones_like(zr))
    v = np.append(vl * np.ones_like(zl), vr * np.ones_like(zr))
    e = p / (r * (gamma - 1))
    m = np.append(rl * dxl * np.ones_like(zl), rr * dxr * np.ones_like(zr))

    for i in range(n_ite - 1):
        r, v, e, z, p, m = update(r, v, e, z, p, m)

    r, v, e, z, p, m = mirror(r, v, e, z, p, m)
    r = sumden(z, m)
    p = (gamma - 1) * r * e

    dv, de = rhs(r, v, z, p, m)

    v = v + dv * dt
    e = e + de * dt

    v_mov = vel(r, v, z, 0.5, m)
    z = z + v_mov * dt

    r, v, e, z, p, m = scrap(r, v, e, z, p, m)

    return r, v, e, z, p
