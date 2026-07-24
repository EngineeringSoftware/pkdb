import math

import cupy as cp
import pykokkos as pk


# Kernel for electron movement and collisions
@pk.workunit
def electron_kernel_1D(
    tid: int,
    Nc: int,
    d_tosum_ar: pk.View2D[float],
    d_data_ar: pk.View2D[float],
    d_collct_ar: pk.View2D[float],
    d_E_ar: pk.View1D[float],
    d_ns_ar: pk.View1D[float],
    d_R_vec: pk.View2D[float],
    d_currxbins: pk.View1D[pk.int64],
    d_forE_xbins: pk.View1D[pk.int64],
    nn: float,
    dt: float,
    d_curr_count: pk.View1D[pk.int64],
    d_g0_ar: pk.View1D[float],
    d_g1_ar: pk.View1D[float],
    d_g2_ar: pk.View1D[float],
    d_g3_ar: pk.View1D[float],
    stride: int,
    zero_g2: bool,
):
    L: float = 2.54
    q_e: float = 1.602 * 10 ** (-19)  # charge of electron (C)
    m_e: float = 9.10938 * 10 ** (-31)  # mass of electron (kg)
    m_n: float = 6.6 * 10 ** (-26)  # mass of neutral particles (argon)
    j_ev: float = 6.241509 * (10**18)  # joules to eV conversion factor

    # Collision energy losses (eV)
    e_ion_full: float = 15.76
    e_exc: float = 11.55
    e_ion_step: float = 4.2

    xx: int = 0
    xy: int = 1
    xz: int = 2
    vx: int = 3
    vy: int = 4
    vz: int = 5

    wt: int = 0
    ai: int = 1
    ae: int = 2
    en: int = 3

    epsilon: float = 0.01

    # Looping and doing all the particles
    for i in range(tid, Nc, stride):
        d_w: float = d_tosum_ar[i][wt]
        nz_ind: int = d_w > 0

        d_xx: float = d_data_ar[i][xx]
        d_xy: float = d_data_ar[i][xy]
        d_xz: float = d_data_ar[i][xz]

        d_vx: float = d_data_ar[i][vx] * nz_ind + epsilon * (1 - nz_ind)
        d_vy: float = d_data_ar[i][vy] * nz_ind + epsilon * (1 - nz_ind)
        d_vz: float = d_data_ar[i][vz] * nz_ind + epsilon * (1 - nz_ind)

        # Energy / velocity of particle
        e_el: float = d_vx**2 + d_vy**2 + d_vz**2
        v_mag: float = math.sqrt(e_el)
        v_inc_x: float = d_vx / v_mag
        v_inc_y: float = d_vy / v_mag
        v_inc_z: float = d_vz / v_mag
        e_el = 0.5 * m_e * j_ev * e_el
        log_e_el: float = math.log10(e_el)

        ## G0: Elastic
        a1: float = -0.02704763
        b1: float = -0.23720051
        c1: float = -19.67900951

        a2: float = -0.08847237
        b2: float = -0.6084786
        c2: float = -20.24111992

        a3: float = -0.37608274
        b3: float = -1.8167778
        c3: float = -21.51414308

        a4: float = -1.4874467
        b4: float = -4.82420619
        c4: float = -23.55745478

        a5: float = -0.9870356
        b5: float = -4.2206026
        c5: float = -23.44715988

        a6: float = 14.28063581
        b6: float = 18.92275458
        c6: float = -14.70556113

        a7: float = -2.12069169
        b7: float = 0.7555105
        c7: float = -19.71943671

        a8: float = -0.3585636
        b8: float = 0.79246666
        c8: float = -19.71260274

        a9: float = 1.25262128
        b9: float = -0.40002029
        c9: float = -19.48909003

        a10: float = -2.28332905
        b10: float = 4.89566076
        c10: float = -21.46789173

        a11: float = -1.47508661
        b11: float = 2.82263476
        c11: float = -20.1928456

        i1: float = 0.000780396
        i2: float = 0.006098667
        i3: float = 0.0476269
        i4: float = 0.1015574
        i5: float = 0.1943527
        i6: float = 0.2995781
        i7: float = 1.097156
        i8: float = 2.339524
        i9: float = 4.988691
        i10: float = 10.63765
        log_sig_g0: float = (
            (e_el < i1) * (((a1 * log_e_el + b1) * log_e_el + c1))
            + (i1 <= e_el and e_el < i2) * (((a2 * log_e_el + b2) * log_e_el + c2))
            + (i2 <= e_el and e_el < i3) * (((a3 * log_e_el + b3) * log_e_el + c3))
            + (i3 <= e_el and e_el < i4) * (((a4 * log_e_el + b4) * log_e_el + c4))
            + (i4 <= e_el and e_el < i5) * (((a5 * log_e_el + b5) * log_e_el + c5))
            + (i5 <= e_el and e_el < i6) * (((a6 * log_e_el + b6) * log_e_el + c6))
            + (i6 <= e_el and e_el < i7) * (((a7 * log_e_el + b7) * log_e_el + c7))
            + (i7 <= e_el and e_el < i8) * (((a8 * log_e_el + b8) * log_e_el + c8))
            + (i8 <= e_el and e_el < i9) * (((a9 * log_e_el + b9) * log_e_el + c9))
            + (i9 <= e_el and e_el < i10) * (((a10 * log_e_el + b10) * log_e_el + c10))
            + (i10 <= e_el) * (((a11 * log_e_el + b11) * log_e_el + c11))
        )
        sig_g0: float = 10**log_sig_g0

        ## G2: IONIZATION
        a1 = -2149239.99337822
        b1 = 5150451.29014456
        c1 = -3085665.64874862
        a2 = -227229.83440255
        b2 = 545171.14839945
        c2 = -327016.69508374
        a3 = -20173.68563579
        b3 = 48599.13254342
        c3 = -29290.96100603
        a4 = -1838.90888364
        b4 = 4491.58838596
        c4 = -2763.82866252
        a5 = -186.98923927
        b5 = 477.04784782
        c5 = -324.75732578
        a6 = -24.48843901
        b6 = 69.43965607
        c6 = -69.145292411
        a7 = -4.33101714
        b7 = 14.47124519
        c7 = -31.66224458
        a8 = -0.68345667
        b8 = 2.6053539
        c8 = -21.99823511

        i1 = 15.77742
        i2 = 15.81964
        i3 = 15.96196
        i4 = 16.44162
        i5 = 18.05837
        i6 = 23.50771
        i7 = 41.87498

        log_sig_g2: float = (
            (e_ion_full < e_el and e_el < i1) * (((a1 * log_e_el + b1) * log_e_el + c1))
            + (i1 <= e_el and e_el < i2) * (((a2 * log_e_el + b2) * log_e_el + c2))
            + (i2 <= e_el and e_el < i3) * (((a3 * log_e_el + b3) * log_e_el + c3))
            + (i3 <= e_el and e_el < i4) * (((a4 * log_e_el + b4) * log_e_el + c4))
            + (i4 <= e_el and e_el < i5) * (((a5 * log_e_el + b5) * log_e_el + c5))
            + (i5 <= e_el and e_el < i6) * (((a6 * log_e_el + b6) * log_e_el + c6))
            + (i6 <= e_el and e_el < i7) * (((a7 * log_e_el + b7) * log_e_el + c7))
            + (i7 <= e_el) * (((a8 * log_e_el + b8) * log_e_el + c8))
        )
        sig_g2: float = 10**log_sig_g2
        if e_el < e_ion_full:
            sig_g2 = 0.0

        sig_g1: float = 0.0
        sig_g3: float = 0.0
        # Scale by heavy density
        sig_g0 *= nn
        sig_g1 *= nn
        sig_g2 *= nn
        sig_g3 *= 1e6 * d_ns_ar[d_currxbins[i]]

        sig_g1 = 0.0
        sig_g3 = 0.0

        # Currently only elastic + ionization
        if zero_g2:
            sig_g2 = 0.0  # For stable results to check correctness
        sig_tot: float = sig_g0 + sig_g1 + sig_g2 + sig_g3
        P_coll: float = 1 - math.exp(-dt * v_mag * sig_tot)

        # Determine which collision occurs
        pcst: float = P_coll * (1.0 / sig_tot)
        sig_range_g0: float = sig_g0 * pcst
        sig_range_g1: float = sig_range_g0 + sig_g1 * pcst
        sig_range_g2: float = sig_range_g1 + sig_g2 * pcst
        sig_range_g3: float = sig_range_g2 + sig_g3 * pcst

        # For grouping g2 and g3
        coll_indicator_g2: bool = (
            sig_range_g1 < d_R_vec[i][0] and d_R_vec[i][0] < sig_range_g2
        )
        coll_indicator_g3: bool = (
            sig_range_g2 < d_R_vec[i][0] and d_R_vec[i][0] < sig_range_g3
        )

        ## G0: ELASTIC
        if d_R_vec[i][0] < sig_range_g0:
            ## Original electron deflection direction (vscat)
            cos_chi: float = 2 * d_R_vec[i][1] - 1
            chi: float = math.acos(cos_chi)
            phi: float = 2 * math.pi * d_R_vec[i][2]
            theta: float = math.acos(v_inc_x)
            sign_sintheta_g0: float = (d_R_vec[i][4] > 0.5) * 1 + (
                d_R_vec[i][4] < 0.5
            ) * (-1)
            # TERM 1
            v_scat_x: float = cos_chi * v_inc_x
            v_scat_y: float = cos_chi * v_inc_y
            v_scat_z: float = cos_chi * v_inc_z
            # TERM 2
            fac: float = (
                math.sin(chi) * math.sin(phi) / (math.sin(theta) * sign_sintheta_g0)
            )
            v_scat_x += 0
            v_scat_y += v_inc_z * fac
            v_scat_z -= v_inc_y * fac
            # TERM 3
            fac = math.sin(chi) * math.cos(phi) / (math.sin(theta) * sign_sintheta_g0)
            v_scat_x += fac * (v_inc_y**2 + v_inc_z**2)
            v_scat_y -= fac * (v_inc_x * v_inc_y)
            v_scat_z -= fac * (v_inc_x * v_inc_z)

            v_mag_new: float = v_mag * math.sqrt(1 - (2 * m_e / m_n) * (1 - cos_chi))
            d_vx = v_scat_x * v_mag_new
            d_vy = v_scat_y * v_mag_new
            d_vz = v_scat_z * v_mag_new

            d_g0_ar[i] = 1.0
            d_collct_ar[i][0] = 1.0

        ## G1: EXCITATION
        elif sig_range_g0 < d_R_vec[i][0] and d_R_vec[i][0] < sig_range_g1:
            ## Original electron deflection direction (vscat)
            cos_chi: float = 2 * d_R_vec[i][1] - 1
            chi: float = math.acos(cos_chi)
            phi: float = 2 * math.pi * d_R_vec[i][2]
            theta: float = math.acos(v_inc_x)
            sign_sintheta_g0: float = (d_R_vec[i][4] > 0.5) * 1 + (
                d_R_vec[i][4] < 0.5
            ) * (-1)
            # TERM 1
            v_scat_x: float = cos_chi * v_inc_x
            v_scat_y: float = cos_chi * v_inc_y
            v_scat_z: float = cos_chi * v_inc_z
            # TERM 2
            fac: float = (
                math.sin(chi) * math.sin(phi) / (math.sin(theta) * sign_sintheta_g0)
            )
            v_scat_x += 0
            v_scat_y += v_inc_z * fac
            v_scat_z -= v_inc_y * fac
            # TERM 3
            fac = math.sin(chi) * math.cos(phi) / (math.sin(theta) * sign_sintheta_g0)
            v_scat_x += fac * (v_inc_y**2 + v_inc_z**2)
            v_scat_y -= fac * (v_inc_x * v_inc_y)
            v_scat_z -= fac * (v_inc_x * v_inc_z)

            v_mag_new: float = math.sqrt(2 * (e_el - e_exc) / (m_e * j_ev))
            d_vx = v_scat_x * v_mag_new
            d_vy = v_scat_y * v_mag_new
            d_vz = v_scat_z * v_mag_new
            d_tosum_ar[i][ae] = 1.0
            d_g1_ar[i] = 1.0
            d_collct_ar[i][1] = 1.0

        ## G2 & G3: IONIZATION & STEPWISE
        elif coll_indicator_g2 or coll_indicator_g3:
            ## Original electron deflection direction (vscat)
            cos_chi: float = 2 * d_R_vec[i][1] - 1
            chi: float = math.acos(cos_chi)
            phi: float = 2 * math.pi * d_R_vec[i][2]
            theta: float = math.acos(v_inc_x)
            sign_sintheta_g0: float = (d_R_vec[i][4] > 0.5) * 1 + (
                d_R_vec[i][4] < 0.5
            ) * (-1)
            # TERM 1
            v_scat_x: float = cos_chi * v_inc_x
            v_scat_y: float = cos_chi * v_inc_y
            v_scat_z: float = cos_chi * v_inc_z
            # TERM 2
            fac: float = (
                math.sin(chi) * math.sin(phi) / (math.sin(theta) * sign_sintheta_g0)
            )
            v_scat_x += 0
            v_scat_y += v_inc_z * fac
            v_scat_z -= v_inc_y * fac
            # TERM 3
            fac = math.sin(chi) * math.cos(phi) / (math.sin(theta) * sign_sintheta_g0)
            v_scat_x += fac * (v_inc_y**2 + v_inc_z**2)
            v_scat_y -= fac * (v_inc_x * v_inc_y)
            v_scat_z -= fac * (v_inc_x * v_inc_z)

            write_ind: int = pk.atomic_fetch_add(d_curr_count, [0], 1)
            write_ind += Nc

            # Energy splitting
            e_ej: float = coll_indicator_g2 * abs(
                0.5 * (e_el - e_ion_full)
            ) + coll_indicator_g3 * abs(0.5 * (e_el - e_ion_step))
            e_scat: float = e_ej

            ## Original electron speed and true velocity vector (uses direction from original calc)
            v_mag_new: float = math.sqrt(2 * e_scat / (j_ev * m_e))

            d_vx = v_scat_x * v_mag_new
            d_vy = v_scat_y * v_mag_new
            d_vz = v_scat_z * v_mag_new

            d_tosum_ar[i][ai] = 1.0
            if coll_indicator_g2:
                d_collct_ar[i][2] = 1.0
                d_g2_ar[i] = 1.0
            d_tosum_ar[i][ae] = coll_indicator_g3 * (-1.0)
            if coll_indicator_g3:
                d_collct_ar[i][3] = 1.0
                d_g3_ar[i] = 1.0

            # Ejected particle exit angle
            cos_chi = 2 * d_R_vec[i][3] - 1
            chi = math.acos(cos_chi)
            phi = 2 * math.pi * d_R_vec[i][5]
            theta = math.acos(v_inc_x)
            sign_sintheta_g2: float = (d_R_vec[i][6] > 0.5) * (1.0) + (
                d_R_vec[i][6] < 0.5
            ) * (-1.0)
            # TERM 1
            v_scat_x = cos_chi * v_inc_x
            v_scat_y = cos_chi * v_inc_y
            v_scat_z = cos_chi * v_inc_z
            # TERM 2
            fac = math.sin(chi) * math.sin(phi) / (math.sin(theta) * sign_sintheta_g2)
            v_scat_x += 0
            v_scat_y += v_inc_z * fac
            v_scat_z -= v_inc_y * fac
            # TERM 3
            fac = math.sin(chi) * math.cos(phi) / (math.sin(theta) * sign_sintheta_g2)
            v_scat_x += fac * (v_inc_y * v_inc_y + v_inc_z * v_inc_z)
            v_scat_y -= fac * (v_inc_x * v_inc_y)
            v_scat_z -= fac * (v_inc_x * v_inc_z)

            # Write new particle data
            d_data_ar[write_ind][vx] = v_scat_x * v_mag_new
            d_data_ar[write_ind][vy] = v_scat_y * v_mag_new
            d_data_ar[write_ind][vz] = v_scat_z * v_mag_new
            d_data_ar[write_ind][xx] = d_xx
            d_data_ar[write_ind][xy] = d_xy
            d_data_ar[write_ind][xz] = d_xz
            d_tosum_ar[write_ind][wt] = d_tosum_ar[i][wt]
            en_x: float = (v_scat_x * v_mag_new) * (v_scat_x * v_mag_new)
            en_y: float = (v_scat_y * v_mag_new) * (v_scat_y * v_mag_new)
            en_z: float = (v_scat_z * v_mag_new) * (v_scat_z * v_mag_new)
            d_tosum_ar[write_ind][en] = 0.5 * m_e * j_ev * (en_x + en_y + en_z)

        ## X-ADV
        d_xx += 100 * dt * d_vx
        d_xy += 100 * dt * d_vy
        d_xz += 100 * dt * d_vz

        # V_ADV
        index: pk.int64 = d_forE_xbins[i]
        d_vx -= dt * (q_e / m_e) * (100 * d_E_ar[index])

        ## BOUNDARY CONDITIONS
        is_valid: int = 0 < d_xx and d_xx < L and d_w > 0
        d_tosum_ar[i][wt] = (is_valid) * d_w

        d_data_ar[i][
            xx
        ] = d_xx  # don't kill for cuda reduction purposes with empty spot (long story)
        d_data_ar[i][xy] = (is_valid) * d_xy
        d_data_ar[i][xz] = (is_valid) * d_xz

        d_vx = (is_valid) * d_vx
        d_vy = (is_valid) * d_vy
        d_vz = (is_valid) * d_vz

        # For temperature calculations
        en_x: float = d_vx * d_vx
        en_y: float = d_vy * d_vy
        en_z: float = d_vz * d_vz
        d_tosum_ar[i][en] = 0.5 * m_e * j_ev * (en_x + en_y + en_z)

        # Putting the v back in
        d_data_ar[i][vx] = d_vx
        d_data_ar[i][vy] = d_vy
        d_data_ar[i][vz] = d_vz


def electron_kernel(
    Nc: int,
    d_data_ar: cp.ndarray,
    d_tosum_ar: cp.ndarray,
    d_collct_ar: cp.ndarray,
    d_E_ar: cp.ndarray,
    d_ns_ar: cp.ndarray,
    d_R_vec: cp.ndarray,
    d_currxbins: cp.ndarray,
    d_forE_xbins: cp.ndarray,
    nn: float,
    dt: float,
    d_curr_count: cp.ndarray,
    num_blocks: int,
    threads_per_block: int,
    d_g0_ar,
    d_g1_ar,
    d_g2_ar,
    d_g3_ar,
    d_coll_indicator_g0_ar,
    # d_coll_indicator_g1_ar,
    # d_coll_indicator_g2_ar,
    # d_coll_indicator_g3_ar,
    zero_g2: bool,
):
    num_threads: int = num_blocks * threads_per_block

    pk.parallel_for(
        "electron_kernel",
        num_threads,
        electron_kernel_1D,
        Nc=Nc,
        d_tosum_ar=d_tosum_ar,
        d_collct_ar=d_collct_ar,
        d_data_ar=d_data_ar,
        d_ns_ar=d_ns_ar,
        d_E_ar=d_E_ar,
        d_R_vec=d_R_vec,
        d_currxbins=d_currxbins,
        d_forE_xbins=d_forE_xbins,
        nn=nn,
        dt=dt,
        d_curr_count=d_curr_count,
        d_g0_ar=d_g0_ar,
        d_g1_ar=d_g1_ar,
        d_g2_ar=d_g2_ar,
        d_g3_ar=d_g3_ar,
        stride=num_threads,
        zero_g2=zero_g2,
    )
    cp.cuda.runtime.deviceSynchronize()
