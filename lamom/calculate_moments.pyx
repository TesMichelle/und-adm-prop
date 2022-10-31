# cython: language_level=3

cimport cython
import numpy as np
from scipy import integrate

from libc.math cimport exp, abs, pow, sqrt


def getk1(double sg, double dur):
    return 1 - pow(1 - sg, dur)

def get_prop_per_gen(double k1, double dur):
    return 1 - pow(1 - k1, 1/dur)

def getk2(s, gs, dur, L=1, N=1000):
    return integrate.dblquad(fun, 0, L, lambda x: x, lambda x: L, args=[s, gs, dur, N])[0]*(1-1/2/N)**gs * 2 / L / L


def getk3(s, gs, dur, L=1, N=1000):

    cdef double[:, ::1] L_matrix = np.array([
            [4*N*N, 0, 0, 0, 0],
            [2*N, 2*N*(2*N-1), 0, 0, 0],
            [2*N, 0, 2*N*(2*N-1), 0, 0],
            [2*N, 0, 0, 2*N*(2*N-1), 0],
            [1, 2*N-1, 2*N-1, 2*N-1, (2*N-1)*(2*N-2)]
        ]) / 4 / N / N

    D = np.array([
        [1 - s, 0, 0, 0, 0],
        [0, pow(1-s, 2), 0, 0, 0],
        [0, 0, pow(1-s, 2), 0, 0],
        [0, 0, 0, pow(1-s, 2), 0],
        [0, 0, 0, 0, pow(1-s, 3)]
    ])

    v0 = np.array(
        [1-s, pow(1-s, 2), pow(1-s, 2), pow(1-s, 2), pow(1-s, 3)]
    )

    return -integrate.tplquad(
        fun3, 0, L,
        lambda x: x, lambda x: L, lambda x, y: y, lambda x, y: L,
        args=[gs, dur, L_matrix, D, v0])[0]*pow((1-1/2/N)*(1-2/2/N), gs) / L / L / L * 6


cdef double geta(double N, double ll, double ldl, double sg):
    return ll*(1-sg)

cdef double getb(double N, double ll, double ldl, double sg):
    return ldl*pow(1-sg, 2)

@cython.cdivision(True)
cdef double getc(double N, double ll, double ldl, double sg):
    return ll*(1-sg)/2./N

@cython.cdivision(True)
cdef double getd(double N, double ll, double ldl, double sg):
    return (1 - 1/2./N + ldl/2./N)*(1-sg)**2

cdef double getdiscr(double a, double b, double c, double d):
    return sqrt(a*a + 4*b*c-2*a*d + d*d)

cdef double getroot1(double a, double b, double c, double d):
    return 0.5*(a+d-getdiscr(a, b, c, d))

cdef double getroot2(double a, double b, double c, double d):
    return 0.5*(a+d+getdiscr(a, b, c, d))

@cython.cdivision(True)
cdef double getvn2(double sg, double g, double length, double N):

    cdef double n = g

    cdef double ll = (1 + exp(-2*length))/2
    cdef double ldl = (1 - exp(-2*length))/2

    cdef double a = geta(N, ll, ldl, sg)
    cdef double b = getb(N, ll, ldl, sg)
    cdef double c = getc(N, ll, ldl, sg)
    cdef double d = getd(N, ll, ldl, sg)

    cdef double discr = getdiscr(a, b, c, d)

    cdef double v00 = 1-sg
    cdef double v01 = (1-sg)*(1-sg)

    cdef double r1 = getroot1(a, b, c, d)
    cdef double r2 = getroot2(a, b, c, d)
    cdef double dr = pow(r1, n) - pow(r2, n)
    return 0.5*(-2*c*(dr)*v00+((a-d)*dr+discr*(r1**n+r2**n))*v01)/ discr

@cython.cdivision(True)
cdef double getvn1(double sg, double g, double length, double N):

    cdef double n = g

    cdef double ll = (1 + exp(-2*length))/2
    cdef double ldl = (1 - exp(-2*length))/2

    cdef double a = geta(N, ll, ldl, sg)
    cdef double b = getb(N, ll, ldl, sg)
    cdef double c = getc(N, ll, ldl, sg)
    cdef double d = getd(N, ll, ldl, sg)

    cdef double discr = getdiscr(a, b, c, d)

    cdef double v00 = 1-sg
    cdef double v01 = (1-sg)*(1-sg)

    r1 = getroot1(a, b, c, d)
    r2 = getroot2(a, b, c, d)
    dr = r1**n - r2**n
    return 0.5 * (discr*v00*(r1**n+r2**n) + (d - a)*dr*v00 - 2*b*v01*dr) / discr


cdef double fun(double l, double ls, double s, double gs, double dur, double N):
    length = l - ls
    return (getvn1(s, dur-1, length, N) - getvn2(s, dur-1, length, N)) * pow(0.5 * (1 + exp(-2*length)), gs)
# cdef double fun(double l, double ls, double s, double gs, double dur):
#     length = abs(l - ls)
#     return (getvn1(s, dur, length) - getvn2(s, dur, length)) * pow(0.5 * (1 + exp(-2*length)), gs)


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def fun3(double l, double ls, double lss,
         double gs, double dur,
         double[:, ::1] L, D, v0):
    # check all these probabilities
    cdef double d1 = abs(ls - l)
    cdef double d2 = abs(lss - l)
    cdef double d3 = abs(lss - ls)

    # Haldane map function
    cdef double lls = (1 + exp(-2*d1)) / 2
    cdef double ldls = (1 - exp(-2*d1)) / 2
    cdef double lsdlss = (1 - exp(-2*d3)) / 2
    cdef double lslss = (1 + exp(-2*d3)) / 2
#     THE SAME - ну и треш
#     cdef double llslss = exp(-2*d2) + (1-exp(-2*d1))/2*exp(-2*d3) + exp(-2*d1)*(1-exp(-2*d3))/2 + (1 - exp(-2*d1))*(1 - exp(-2*d3))/4
#     cdef double llsdlss = exp(-2*d1) * (1 - exp(-2*d3)) / 2 + (1 - exp(-2*d1))*(1 - exp(-2*d3))/4
#     cdef double ldlslss = (1 - exp(-2*d1)) / 2 * exp(-2*d3) + (1 - exp(-2*d1))*(1 - exp(-2*d3))/4
#     cdef double ldlsdlss = ldls*lsdlss
    cdef double llslss = lls*lslss
    cdef double llsdlss = lls*lsdlss
    cdef double ldlslss = ldls*lslss
    cdef double ldlsdlss = ldls*lsdlss
    cdef double llss = llslss + ldlsdlss
    cdef double ldlss = 1 - (llslss + ldls*lsdlss)

    # my
    U = np.array([
        [llslss, ldlslss, ldlsdlss, llsdlss, 0],
        [0, lslss, 0, 0, lsdlss],
        [0, 0, llss, 0, ldlss],
        [0, 0, 0, lls, ldls],
        [0, 0, 0, 0, 1]
    ])

#     # using the same map as in LaNeta
#     cdef double lls = (exp(-d1))
#     cdef double ldls = (1 - exp(-d1))
#     cdef double lsdlss = (1 - exp(-d3))
#     cdef double lslss = (exp(-d3))
# #     THE SAME - ну и треш
# #     cdef double llslss = exp(-2*d2) + (1-exp(-2*d1))/2*exp(-2*d3) + exp(-2*d1)*(1-exp(-2*d3))/2 + (1 - exp(-2*d1))*(1 - exp(-2*d3))/4
# #     cdef double llsdlss = exp(-2*d1) * (1 - exp(-2*d3)) / 2 + (1 - exp(-2*d1))*(1 - exp(-2*d3))/4
# #     cdef double ldlslss = (1 - exp(-2*d1)) / 2 * exp(-2*d3) + (1 - exp(-2*d1))*(1 - exp(-2*d3))/4
# #     cdef double ldlsdlss = ldls*lsdlss
#     cdef double llslss = lls*lslss
#     cdef double llsdlss = lls*lsdlss
#     cdef double ldlslss = ldls*lslss
#     cdef double ldlsdlss = ldls*lsdlss
#     cdef double llss = llslss + ldlsdlss
#     cdef double ldlss = 1 - (llslss + ldls*lsdlss)

    # # preprint
    # U = np.array([
    # [llslss, llsdlss, ldlslss, ldlsdlss, 0],
    # [0, lls, 0, 0, ldls],
    # [0, 0, lslss, 0, ldlss],
    # [0, 0, 0, llss, lsdlss],
    # [0, 0, 0, 0, 1]
    # ])
    eigv = np.array([1, -1, -1, -1, 2], dtype=float)

    A = L @ U @ D
    U, Q = np.linalg.eig(A)
    U = np.diag(U**(dur-1))
    Q_inv = np.linalg.solve(Q, np.eye(5))
    return  (eigv @ Q) @ U @ (Q_inv @ v0) * pow(llslss, gs)
