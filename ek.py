# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 14:22:20 2017

@author: B. Font Garcia
@style: PEP8

@description:
Script to calculate the energy spectra E = E(k) from a one- two- or three-dimensional velocity vector field.
Included there is a function to load binary data and a function to generate and save the plot which
can be helpful.

The energy spectra is calculated in the Fourier space, so a Fourier expansion of the velocity components
is required. Therefore, the data set (velocity) must come from a uniform grid.

Change the constants to whatever suits your computational domain, set the desired spectral integration
bandwidth (kres) and your file containing the velocity data.

@contact: b.fontgarcia@soton.ac.uk
"""

# Imports
import numpy as np
# ! Uncomment for running plyplot without window display
# import matplotlib
# matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import pickle

# Constants
N = 1536                    # Number of grid points in i direction
M = 512                     # Number of grid points in j direction
L = 640                     # Number of grid points in k direction
D = 64                      # Characteristic length value

xmin, xmax = -256, 1280     # Domain size in i direction
ymin, ymax = -256, 256      # Domain size in j direction
zmin, zmax = 0, 641         # Domain size in k direction

kres = 500                  # Resolution of the spectral bandwith integration: dk = (kmax-kmin)/kres

# File containig velocity field
file = '/media/bfg/SSD/Workspace/Lotus/projects/cylinder/3d/second_set/d64_uni/output/6400_output.dat'


# Internal functions
def wavenumbers(dim, **kwargs):
    """
    Return the one- two- or three-dimensional wavenumber vector depending on the 'dim' argument
    for a uniform grid defined in [xmax-xmin, ymax-ymin, zmax-zmin] containing [N, M, L] grid points.

    For the one- and two-dimensional cases, an axis needs to be specified e.g. wavenumbers(dim=2, axis=3)
    will return the two-dimensional wavenumber vector k = (kx, ky) of a 2D domain [xmax-xmin, ymax-ymin]
    """
    # Get the axis argument (only requiered for 1D and 2D)
    axis = kwargs.get('axis', None)

    # Calculate the basic wavenumbers
    alpha = 2*np.pi/(xmax-xmin)
    beta = 2*np.pi/(ymax-ymin)
    gamma = 2*np.pi/(zmax-zmin)

    # Calculate the index per wavenumber direction: eg x: 0,1,2,...,N/2-1,-N/2,-N/2+1,...,-1
    x_index = np.fft.fftfreq(N, d=1/N)
    y_index = np.fft.fftfreq(M, d=1/M)
    z_index = np.fft.fftfreq(L, d=1/L)

    # Initialize arrays
    kx = np.zeros(N)
    ky = np.zeros(M)
    kz = np.zeros(L)

    # Calculate wavenumber vector components: kx, ky, kz
    if dim == 1:
        if axis == 1:
            for i in range(0, N):
                kx[i] = alpha*x_index[i]
            return(kx)
        elif axis == 2:
            for j in range(0, M):
                ky[j] = beta*y_index[j]
            return(ky)
        elif axis == 3:
            for k in range(0, L):
                kz[k] = gamma*z_index[k]
            return(kz)
        else:
            return -1
    elif dim == 2:
        if axis == 1:
            for j in range(0, M):
                ky[j] = beta*y_index[j]
            for k in range(0, L):
                kz[k] = gamma*z_index[k]
            return(ky, kz)
        elif axis == 2:
            for i in range(0, N):
                kx[i] = alpha*x_index[i]
            for k in range(0, L):
                kz[k] = gamma*z_index[k]
            return(kx, kz)
        elif axis == 3:
            for i in range(0, N):
                kx[i] = alpha*x_index[i]
            for j in range(0, M):
                ky[j] = beta*y_index[j]
            return(kx, ky)
        else:
            return -1
    elif dim == 3:
        for i in range(0, N):
            kx[i] = alpha*x_index[i]
        for j in range(0, M):
            ky[j] = beta*y_index[j]
        for k in range(0, L):
            kz[k] = gamma*z_index[k]
        return(kx, ky, kz)
    else:
        return -1


def KE1D(uk, kx):
    """
    Return kinetic energy (KE) of a one-dimensional velocity vector field in Fourier space (uk)
    given the wavenumber kx.

    First, the kinetic energy is computed at each possible wavenumber in Fourier space and stored
    in the E_entries[total number of entries, columns for k and E] array.

    Then, a loop through the entries stores the energy at the corresponging wavenumber bandwidth
    kmod - dk/2 <= kmod <= kmod + dk/2, where dk is calculated acording to the desired resolution kres.
    """
    # Get shape of the input arrays
    n = uk.shape[0]
    if uk.shape != kx.shape:
        return -1

    # Calculate the E_entries array
    E_entries = np.zeros((n, 2))
    for i in range(0, n):
        kmod = np.abs(kx[i])
        a = uk[i]*uk[i].conjugate()
        E = 0.5*(a.real)
        E_entries[i, :] = [kmod, E]

    # Integrate the energy corresponding to a certain bandwidth dk
    kmin = 0
    kmax = np.max(kx)
    dk = (kmax-kmin)/kres
    KE = np.zeros((kres, 2))
    KE[:, 0] = np.linspace(0, kres-1, kres)*dk+dk/2  # k values at half of each bandwidth
    for i in range(0, n):
        kmod = E_entries[i, 0]
        kint = int(kmod/dk)
        if kint >= kres:
            KE[-1, 1] = KE[-1, 1] + E_entries[i, 1]
        else:
            KE[kint, 1] = KE[kint, 1] + E_entries[i, 1]
    return KE


def KE2D(uk, vk, kx, ky):
    """
    Return kinetic energy (KE) of a two-dimensional velocity vector field in Fourier space (uk, vk)
    given the wavenumber vector (kx, ky).

    First, the kinetic energy is computed at each possible wavenumber in Fourier space and stored
    in the E_entries[total number of entries, columns for k and E] array. Note that the wavenumber
    kmod might be repeated during the procedure, so this is why the energy is stored as a set of entries.

    Then, a loop through the entries stores the energy at the corresponging wavenumber bandwidth
    kmod - dk/2 <= kmod <= kmod + dk/2, where dk is calculated acording to the desired resolution kres.
    """
    # Get shape of the input arrays
    n = uk.shape[0]
    m = uk.shape[1]
    if uk.shape != vk.shape:
        return -1

    # Calculate the E_entries array
    E_entries = np.zeros((n*m, 2))
    for i in range(0, n):
        for j in range(0, m):
            kmod = np.sqrt(kx[i]**2+ky[j]**2)
            a = uk[i, j]*uk[i, j].conjugate()
            b = vk[i, j]*vk[i, j].conjugate()
            E = 0.5*(a.real+b.real)
            E_entries[i*m+j, :] = [kmod, E]

    # Integrate the energy corresponding to a certain bandwidth dk
    kmin = 0
    kmax = np.sqrt(np.max(kx)**2+np.max(ky)**2)
    dk = (kmax-kmin)/kres
    KE = np.zeros((kres, 2))
    KE[:, 0] = np.linspace(0, kres-1, kres)*dk+dk/2  # k values at half of each bandwidth
    for i in range(0, n*m):
        kmod = E_entries[i, 0]
        kint = int(kmod/dk)
        if kint >= kres:
            KE[-1, 1] = KE[-1, 1] + E_entries[i, 1]
        else:
            KE[kint, 1] = KE[kint, 1] + E_entries[i, 1]
    return KE


def KE3D(uk, vk, wk, kx, ky, kz):
    """
    Return kinetic energy (KE) of a three-dimensional velocity vector field in Fourier space (uk, vk, wk)
    given the wavenumber vector (kx, ky, kz).

    First, the kinetic energy is computed at each possible wavenumber in Fourier space and stored
    in the E_entries[total number of entries, columns for k and E] array. Note that the wavenumber
    kmod might be repeated during the procedure, so this is why the energy is stored as a set of entries.

    Then, a loop through the entries stores the energy at the corresponging wavenumber bandwidth
    kmod - dk/2 <= kmod <= kmod + dk/2, where dk is calculated acording to the desired resolution kres.
    """
    # Get shape of the input arrays
    n = uk.shape[0]
    m = uk.shape[1]
    l = uk.shape[2]
    if uk.shape != vk.shape or uk.shape != wk.shape or vk.shape != wk.shape:
        return -1

    # Calculate the E_entries array
    E_entries = np.zeros((n*m*l, 2))
    for i in range(0, n):
        for j in range(0, m):
            for k in range(0, l):
                kmod = np.sqrt(kx[i]**2+ky[j]**2+kz[k]**2)
                a = uk[i, j, k]*uk[i, j, k].conjugate()
                b = vk[i, j, k]*vk[i, j, k].conjugate()
                c = wk[i, j, k]*wk[i, j, k].conjugate()
                E = 0.5*(a.real+b.real+c.real)
                E_entries[i*m*l+j*l+k, :] = [kmod, E]

    # Integrate the energy corresponding to a certain bandwidth dk
    kmin = 0
    kmax = np.sqrt(np.max(kx)**2+np.max(ky)**2+np.max(kz)**2)
    dk = (kmax-kmin)/kres
    KE = np.zeros((kres, 2))
    KE[:, 0] = np.linspace(0, kres-1, kres)*dk+dk/2  # k values at half of each bandwidth
    for i in range(0, n*m*l):
        kmod = E_entries[i, 0]
        kint = int(kmod/dk)
        if kint >= kres:
            KE[-1, 1] = KE[-1, 1] + E_entries[i, 1]
        else:
            KE[kint, 1] = KE[kint, 1] + E_entries[i, 1]
    return KE


def plotE(KE, file):
    """
    Generate a loglog plot of the energy spectra using the matplotlib library given the arguments
    KE = [k,E] and the file name to save the plot in .pdf format (can be easly converted to .svg).
    """
    # Basic definitions
    plt.rcParams['text.usetex'] = True  # Set TeX interpreter
    ax = plt.gca()
    fig  = plt.gcf()

    # Generate loglog plot lines
    k = KE[:, 0]*D  # Scale the wavenumber with the characteristic length
    E = KE[:, 1]
    # Show lines
    plt.loglog(k, E, color='black', lw=1.5, label=r'$3\mathrm{D}\,\, \mathrm{total}$')
    plt.loglog([min(k[1:]), max(k)], [0.015*10**(-5/3*np.log10(min(k[1:]))), 0.015*10**(-5/3*np.log10(max(k)))],
               color='black', lw=1, ls='dotted')
    plt.loglog([min(k[1:]), max(k)], [0.015*10**(-3*np.log10(min(k[1:]))), 0.015*10**(-3*np.log10(max(k)))],
               color='black', lw=1, ls='dotted')

    # Set limits
    ax.set_xlim(min(k), max(k))
    ax.set_ylim(1e-12, 1)

    # Make the plot square --------
    fwidth = fig.get_figwidth()
    fheight = fig.get_figheight()
    # get the axis size and position in relative coordinates
    # this gives a BBox object
    bb = ax.get_position()
    # calculate them into real world coordinates
    axwidth = fwidth*(bb.x1-bb.x0)
    axheight = fheight*(bb.y1-bb.y0)
    # if the axis is wider than tall, then it has to be narrowe
    if axwidth > axheight:
        # calculate the narrowing relative to the figure
        narrow_by = (axwidth-axheight)/fwidth
        # move bounding box edges inwards the same amount to give the correct width
        bb.x0 += narrow_by/2
        bb.x1 -= narrow_by/2
    # else if the axis is taller than wide, make it vertically smaller
    # works the same as above
    elif axheight > axwidth:
        shrink_by = (axheight-axwidth)/fheight
        bb.y0 += shrink_by/2
        bb.y1 -= shrink_by/2
    ax.set_position(bb)
    # --------

    # Edit frame, labels and legend
    ax.axhline(linewidth=1)
    ax.axvline(linewidth=1)
    plt.xlabel(r'$\kappa D$')
    plt.ylabel(r'$E(\kappa)$')
    leg = plt.legend(loc='upper right')
    leg.get_frame().set_edgecolor('black')

    # Anotations
    plt.text(x=30, y=1e-4, s=r'$-5/3$', color='black')
    plt.text(x=30, y=1e-7, s=r'$-3$', color='black')

    # Show plot and save figure
    plt.show()
    plt.savefig(file, transparent=True, bbox_inches='tight')
    return


def read_data(file, dim, dtype, stream, periodic):
    """
    Return the velocity components of a velocity vector field stored in binary format.
    The data field is suposed to have been written as: (for k; for j; for i;) where the last dimension
    is the quickest varying index. Each record should have been written as: u, v, w.
    The return velocity components are always converted in np.double precision type.

    Args:
        dim: number of dimensions of the velocity vector field.
        dtype: numpy dtype object. Single or double precision expected.
        stream: type of access of the binary output. If true, only a pure binary output of the velocity
            vector field is assumed. If false, there is a 4-byte header and footer around each "record"
            in the binary file (can happen in some Fortran compilers if access != 'stream').
        periodic: If the user desires to make the data spanwise periodic (true) or not (false).
    """
    if dim == 2:
        if stream:
            shape = (M, N, 2)
            f = open(file, 'rb')
            data = np.fromfile(file=f, dtype=dtype).reshape(shape)
            f.close()
            u = data[:, :, 0].transpose(1, 0)
            v = data[:, :, 1].transpose(1, 0)
            del data
            u = u.astype(np.float64, copy=False)
            v = v.astype(np.float64, copy=False)
            return (u, v)
        else:
            shape = (M, N, 4)
            f = open(file, 'rb')
            data = np.fromfile(file=f, dtype=dtype).reshape(shape)
            f.close()
            u = data[:, :, 1].transpose(1, 0)
            v = data[:, :, 2].transpose(1, 0)
            del data
            u = u.astype(np.float64, copy=False)
            v = v.astype(np.float64, copy=False)
            return (u, v)
    if dim == 3:
        if stream:
            shape = (L, M, N, 3)
            f = open(file, 'rb')
            data = np.fromfile(file=f, dtype=dtype).reshape(shape)
            f.close()
            u = data[:, :, :, 0].transpose(2, 1, 0)
            v = data[:, :, :, 1].transpose(2, 1, 0)
            w = data[:, :, :, 2].transpose(2, 1, 0)
            del data
            if periodic:
                u = np.dstack((u, u[:, :, 0]))
                v = np.dstack((v, v[:, :, 0]))
                w = np.dstack((w, w[:, :, 0]))
            u = u.astype(np.float64, copy=False)
            v = v.astype(np.float64, copy=False)
            w = w.astype(np.float64, copy=False)
            return(u, v, w)    
        else:
            shape = (L, M, N, 5)
            f = open(file, 'rb')
            data = np.fromfile(file=f, dtype=dtype).reshape(shape)
            f.close()
            u = data[:, :, :, 1].transpose(2, 1, 0)
            v = data[:, :, :, 2].transpose(2, 1, 0)
            w = data[:, :, :, 3].transpose(2, 1, 0)
            del data
            # Make the data
            if periodic:
                u = np.dstack((u, u[:, :, 0]))
                v = np.dstack((v, v[:, :, 0]))
                w = np.dstack((w, w[:, :, 0]))
            u = u.astype(np.float64, copy=False)
            v = v.astype(np.float64, copy=False)
            w = w.astype(np.float64, copy=False)
            return(u, v, w)
    else:
        return -1


"""""""""""""""""""""""""""
Main function

The normal steps to generate an energy spectra plot are defined in the main function.

1. Load data: Either used the provided function or any which suits you to load the velocity field.
    I tend to use pickle as is faster if you have already loaded the data in pyhton before.
2. Calculate wavenumber vector: Either 1D, 2D or 3D in any axis you want (which matches the size of the
    velocity field vector).
4. Transoform velocity field from real to Fourier space: You can use numpy.fft which is very useful for this.
    For a 2D field is: uk = np.fft.fft2(u)/(N*M). Note you have to scale it with the total field size N*M.
5. Calculate energy spectra: Use the KE function in the dimensions of your velocity field,
    e.g. KE = KE2D(uk, vk, kx, ky)
6. Plot your energy spectra: Use the function plotE and just pass the KE array and the name of the file,
    e.g. plotE(KE, 'test.pdf')

That's all!

"""""""""""""""""""""""""""


def main():
    global u_avg, v_avg, kx, ky, KE, uk, vk

    with open('u_avg.pickle', 'rb') as f1:
        u_avg = pickle.load(f1)

    with open('v_avg.pickle', 'rb') as f2:
        v_avg = pickle.load(f2)

    uk = np.fft.fft2(u_avg)/(N*M)
    vk = np.fft.fft2(v_avg)/(N*M)
    (kx, ky) = wavenumbers(dim=2, axis=3)
    KE = KE2D(uk, vk, kx, ky)
    plotE(KE, 'test1.pdf')


if __name__ == '__main__':
    main()
