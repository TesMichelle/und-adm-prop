import numpy as np
from scipy import integrate

from numpy import exp, abs, power as pow, sqrt


def getk1(sg, dur):
    return 1 - pow(1 - sg, dur)

def get_prop_per_gen(k1, dur):
    return 1 - pow(1 - k1, 1/dur)

def getk2(s, gs, dur, L=1, N=1000):
    return integrate.dblquad(fun, 0, L, lambda x: x, lambda x: L, args=[s, gs, dur, N])[0]*(1-1/2/N)**gs * 2 / L / L


def getk3(s, gs, dur, L=1, N=1000):

    L_matrix = np.array([
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
        args=[gs-1, dur, N, L_matrix, D, v0])[0]*pow((1-1/2/N)*(1-2/2/N), gs-1) / L / L / L * 6

def geta(N, ll, ldl, sg):
    return ll*(1-sg)

def getb(N, ll, ldl, sg):
    return ldl*(1-sg)

def getc(N, ll, ldl, sg):
    return ll*(1-sg)/2./N


def getd(N, ll, ldl, sg):
    return ldl*(1-sg)/2./N + (1-1/2/N)*(1-sg)**2

def getdiscr(a, b, c, d):
    return sqrt(a*a + 4*b*c-2*a*d + d*d)

def getroot1(a, b, c, d):
    return 0.5*(a+d-getdiscr(a, b, c, d))

def getroot2(a, b, c, d):
    return 0.5*(a+d+getdiscr(a, b, c, d))


def getvn2(sg, g, length, N):

    n = g

    ll = (1 + exp(-2*length))/2
    ldl = (1 - exp(-2*length))/2

    a = geta(N, ll, ldl, sg)
    b = getb(N, ll, ldl, sg)
    c = getc(N, ll, ldl, sg)
    d = getd(N, ll, ldl, sg)

    discr = getdiscr(a, b, c, d)

    v00 = 1-sg
    v01 = (1-sg)*(1-sg)

    r1 = getroot1(a, b, c, d)
    r2 = getroot2(a, b, c, d)
    dr = pow(r1, n) - pow(r2, n)
    return 0.5*(-2*c*(dr)*v00+((a-d)*dr+discr*(r1**n+r2**n))*v01)/ discr


def getvn1(sg, g, length, N):

    n = g

    ll = (1 + exp(-2*length))/2
    ldl = (1 - exp(-2*length))/2

    a = geta(N, ll, ldl, sg)
    b = getb(N, ll, ldl, sg)
    c = getc(N, ll, ldl, sg)
    d = getd(N, ll, ldl, sg)

    discr = getdiscr(a, b, c, d)

    v00 = 1-sg
    v01 = (1-sg)*(1-sg)

    r1 = getroot1(a, b, c, d)
    r2 = getroot2(a, b, c, d)
    dr = r1**n - r2**n
    return 0.5 * (discr*v00*(r1**n+r2**n) + (d - a)*dr*v00 - 2*b*v01*dr) / discr


def fun(l, ls, s, gs, dur, N):
    length = l - ls
    return (getvn1(s, dur-1, length, N) - getvn2(s, dur-1, length, N)) * pow(0.5 * (1 + exp(-2*length)), gs-1)
#   fun( l,  ls,  s,  gs,  dur):
#     length = abs(l - ls)
#     return (getvn1(s, dur, length) - getvn2(s, dur, length)) * pow(0.5 * (1 + exp(-2*length)), gs)


def fun3(l, ls, lss,
         gs, dur, N,
         L, D, v0):
    # check all these probabilities
    d1 = abs(ls - l)
    d2 = abs(lss - l)
    d3 = abs(lss - ls)

    # Haldane map function
    lls = (1 + exp(-2*d1)) / 2
    ldls = (1 - exp(-2*d1)) / 2
    lsdlss = (1 - exp(-2*d3)) / 2
    lslss = (1 + exp(-2*d3)) / 2
#     THE SAME - ну и треш
#       llslss = exp(-2*d2) + (1-exp(-2*d1))/2*exp(-2*d3) + exp(-2*d1)*(1-exp(-2*d3))/2 + (1 - exp(-2*d1))*(1 - exp(-2*d3))/4
#       llsdlss = exp(-2*d1) * (1 - exp(-2*d3)) / 2 + (1 - exp(-2*d1))*(1 - exp(-2*d3))/4
#       ldlslss = (1 - exp(-2*d1)) / 2 * exp(-2*d3) + (1 - exp(-2*d1))*(1 - exp(-2*d3))/4
#       ldlsdlss = ldls*lsdlss
    llslss = lls*lslss
    llsdlss = lls*lsdlss
    ldlslss = ldls*lslss
    ldlsdlss = ldls*lsdlss
    llss = llslss + ldlsdlss
    ldlss = 1 - (llslss + ldls*lsdlss)

    # my
    U = np.array([
        [llslss, llsdlss, ldlslss, ldlsdlss, 0],
        [0, lls, 0, 0, ldls],
        [0, 0, lslss, 0, lsdlss],
        [0, 0, 0, llss, ldlss],
        [0, 0, 0, 0, 1]
    ])

#     # using the same map as in LaNeta
#       lls = (exp(-d1))
#       ldls = (1 - exp(-d1))
#       lsdlss = (1 - exp(-d3))
#       lslss = (exp(-d3))
# #     THE SAME - ну и треш
# #       llslss = exp(-2*d2) + (1-exp(-2*d1))/2*exp(-2*d3) + exp(-2*d1)*(1-exp(-2*d3))/2 + (1 - exp(-2*d1))*(1 - exp(-2*d3))/4
# #       llsdlss = exp(-2*d1) * (1 - exp(-2*d3)) / 2 + (1 - exp(-2*d1))*(1 - exp(-2*d3))/4
# #       ldlslss = (1 - exp(-2*d1)) / 2 * exp(-2*d3) + (1 - exp(-2*d1))*(1 - exp(-2*d3))/4
# #       ldlsdlss = ldls*lsdlss
#       llslss = lls*lslss
#       llsdlss = lls*lsdlss
#       ldlslss = ldls*lslss
#       ldlsdlss = ldls*lsdlss
#       llss = llslss + ldlsdlss
#       ldlss = 1 - (llslss + ldls*lsdlss)

#     # preprint
#     U = np.array([
#     [llslss, llsdlss, ldlslss, ldlsdlss, 0],
#     [0, lls, 0, 0, ldls],
#     [0, 0, lslss, 0, ldlss],
#     [0, 0, 0, llss, lsdlss],
#     [0, 0, 0, 0, 1]
#     ])
    eigv = np.array([1, -1, -1, -1, 2], dtype=float)

    A = L @ D @ U
    U, Q = np.linalg.eig(A)
    U = np.diag(U**(dur-1))
    Q_inv = np.linalg.solve(Q, np.eye(5))
    return  (eigv @ Q) @ U @ (Q_inv @ v0) * pow(llslss, gs)
