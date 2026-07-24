import pykokkos as pk


@pk.workunit
def kernel_i_plus_100(i, a):
    a[i] = i+100
    a[i] +=20
