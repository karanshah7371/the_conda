import arviz as az
import numpy as np
import pymc3 as pm
import pymc3.math as pmm
import matplotlib.pyplot as plt
import aesara.tensor as at

def Prob( MmentsA , MmentsB , TrueState ):
    # Density matrix has form
    # A ReE -iImE ReF -iImF ReG -iImG
    # ReE+ iImE B ReH -iImH ReI -iImI
    # ReF+ iImF ReH + iImH C ReJ -imJ
    # ReG+ iImG ReI + iImI ReJ + imJ D
    A = TrueState [0,0]
    B = TrueState [1,1]
    C = TrueState [2,2]
    D = TrueState [3,3]
    ReE = np.real( TrueState [1,0])
    ImE = np.imag( TrueState [1,0])
    ReF = np.real( TrueState [2,0])
    ImF = np.imag( TrueState [2,0])
    ReG = np.real( TrueState [3,0])
    ImG = np.imag( TrueState [3,0])
    ReH = np.real( TrueState [2,1])
    ImH = np.imag( TrueState [2,1])
    ReI = np.real( TrueState [3,1])
    ImI = np.imag( TrueState [3,1])
    ReJ = np.real( TrueState [3,2])
    ImJ = np.imag( TrueState [3,2])
    # extract measurement settings
    ThetaQA = MmentsA [:,1]
    ThetaQB = MmentsB [:,1]
    ThetaHA = MmentsA [:,0]
    ThetaHB = MmentsB [:,0]
    hA = MmentsA [:,2]
    hB = MmentsB [:,2]
    vA = MmentsA [:,3]
    vB = MmentsB [:,3]
    # measurement probability model
    prob = 1./32.*(8*A*hA*hB+8*B*hA*hB+8*C*hA*hB+8*D*hA*hB+8*A*hB*vA \
    +8*B*hB*vA+8*C*hB*vA+8*D*hB*vA+8*A*hA*vB+8*B*hA*vB+8*C*hA*vB \
    +8*D*hA*vB+8*A*vA*vB+8*B*vA*vB+8*C*vA*vB+8*D*vA*vB \
    +4*(A+B-C-D)*(hA -vA)*(hB+vB)*np.cos(4* ThetaHA ) \
    +(A-B-C+D+2* ReG +2*ReH)*(hA -vA)*(hB -vB)*np.cos(4* ThetaHA -4* ThetaHB ) \
    +4*(A-B+C-D)*(hA+vA)*(hB -vB)*np.cos(4* ThetaHB ) \
    +(A-B-C+D-2* ReG -2*ReH)*(hA -vA)*(hB -vB)*np.cos(4*( ThetaHA + ThetaHB )) \
    +4*(A+B-C-D)*(hA -vA)*(hB+vB)*np.cos(4* ThetaHA -4* ThetaQA ) \
    +(A-B-C+D-2* ReG -2*ReH)*(hA -vA)*(hB -vB)*np.cos(4* ThetaHA -4* ThetaHB -4* ThetaQA ) \
    -4*( ImG + ImH )*(hA -vA)*(hB -vB)*np.cos(4* ThetaHA -4* ThetaHB -2* ThetaQA ) \
    +4*(ImG+ ImH)*(hA -vA)*(hB -vB)*np.cos(4* ThetaHA +4* ThetaHB -2* ThetaQA )\
    +(A-B-C+D+2* ReG +2*ReH)*(hA -vA)*(hB -vB)*np.cos(4*( ThetaHA + ThetaHB -ThetaQA )) \
    +4*(A-B+C-D)*(hA+vA)*(hB -vB)*np.cos(4* ThetaHB -4* ThetaQB ) \
    -4*( ImG + ImH )*(hA -vA)*(hB -vB)*np.cos(4* ThetaHA +4* ThetaHB -2* ThetaQA -4* ThetaQB ) \
    +4*(ImG -ImH)*(hA -vA)*(hB -vB)*np.cos(4* ThetaHA +4* ThetaHB -2* ThetaQB ) \
    -4*( ImG -ImH )*(hA -vA)*(hB -vB)*np.cos(4* ThetaHA +4* ThetaHB -4* ThetaQA -2* ThetaQB ) \
    +(A-B-C+D+2* ReG +2*ReH)*(hA -vA)*(hB -vB)*np.cos(4*( ThetaHA + ThetaHB -ThetaQB )) \
    +(A-B-C+D-2* ReG -2*ReH)*(hA -vA)*(hB -vB)*np.cos(4*( ThetaHA + ThetaHB -ThetaQA -ThetaQB ))\
    +(A-B-C+D-2* ReG -2*ReH)*(hA -vA)*(hB -vB)*np.cos(4*( ThetaHA -ThetaHB + ThetaQB )) \
    +(A-B-C+D+2* ReG +2*ReH)*(hA -vA)*(hB -vB)*np.cos(4*( ThetaHA -ThetaHB -ThetaQA + ThetaQB )) \
    -4*( ImG -ImH )*(hA -vA)*(hB -vB)*np.cos(4* ThetaHA -4* ThetaHB +2* ThetaQB ) \
    +4*(ImG -ImH)*(hA -vA)*(hB -vB)*np.cos(4* ThetaHA -4* ThetaHB -4* ThetaQA +2* ThetaQB ) \
    -8*( ReG -ReH )*(hA -vA)*(hB -vB)*np.cos(4* ThetaHA -4* ThetaHB -2* ThetaQA +2* ThetaQB ) \
    +4*(ImG+ ImH)*(hA -vA)*(hB -vB)*np.cos(4* ThetaHA -4* ThetaHB -2* ThetaQA +4* ThetaQB ) \
    +8*(ReG -ReH)*(hA -vA)*(hB -vB)*np.cos(4* ThetaHA +4* ThetaHB -2*( ThetaQA + ThetaQB )) \
    +8*(ReF+ ReI)*(hA -vA)*(hB+vB)*np.sin(4* ThetaHA ) \
    -2*( ReE -ReF +ReI -ReJ)*(hA -vA)*(hB -vB)*np.sin(4* ThetaHA -4* ThetaHB ) \
    +8*(ReE+ ReJ)*(hA+vA)*(hB -vB)*np.sin(4* ThetaHB ) \
    +2*(ReE+ ReF -ReI -ReJ )*(hA -vA)*(hB -vB)*np.sin(4*( ThetaHA + ThetaHB )) \
    -8*( ReF + ReI )*(hA -vA)*(hB+vB)*np.sin(4* ThetaHA -4* ThetaQA ) \
    -2*( ReE + ReF -ReI -ReJ)*(hA -vA)*(hB -vB)*np.sin(4* ThetaHA -4* ThetaHB -4* ThetaQA ) \
    -16*( ImF + ImI )*(hA -vA)*(hB+vB)*np.sin(4* ThetaHA -2* ThetaQA ) \
    -4*( ImF -ImI )*(hA -vA)*(hB -vB)*np.sin(4* ThetaHA -4* ThetaHB -2* ThetaQA ) \
    -4*( ImF -ImI )*(hA -vA)*(hB -vB)*np.sin(4* ThetaHA +4* ThetaHB -2* ThetaQA ) \
    +2*(ReE -ReF+ ReI -ReJ )*(hA -vA)*(hB -vB)*np.sin(4*( ThetaHA + ThetaHB -ThetaQA )) \
    -8*( ReE + ReJ )*(hA+vA)*(hB -vB)*np.sin(4* ThetaHB -4* ThetaQB ) \
    -4*( ImF -ImI )*(hA -vA)*(hB -vB)*np.sin(4* ThetaHA +4* ThetaHB -2* ThetaQA -4* ThetaQB ) \
    -16*( ImE + ImJ )*(hA+vA)*(hB -vB)*np.sin(4* ThetaHB -2* ThetaQB ) \
    -4*( ImE -ImJ )*(hA -vA)*(hB -vB)*np.sin(4* ThetaHA +4* ThetaHB -2* ThetaQB ) \
    -4*( ImE -ImJ )*(hA -vA)*(hB -vB)*np.sin(4* ThetaHA +4* ThetaHB -4* ThetaQA -2* ThetaQB ) \
    -2*( ReE -ReF +ReI -ReJ)*(hA -vA)*(hB -vB)*np.sin(4*( ThetaHA + ThetaHB -ThetaQB )) \
    -2*( ReE + ReF -ReI -ReJ)*(hA -vA)*(hB -vB)*np.sin(4*( ThetaHA + ThetaHB -ThetaQA -ThetaQB )) \
    +2*(ReE+ ReF -ReI -ReJ )*(hA -vA)*(hB -vB)*np.sin(4*( ThetaHA -ThetaHB + ThetaQB )) \
    +2*(ReE -ReF+ ReI -ReJ )*(hA -vA)*(hB -vB)*np.sin(4*( ThetaHA -ThetaHB -ThetaQA + ThetaQB )) \
    +4*(ImE -ImJ)*(hA -vA)*(hB -vB)*np.sin(4* ThetaHA -4* ThetaHB +2* ThetaQB ) \
    +4*(ImE -ImJ)*(hA -vA)*(hB -vB)*np.sin(4* ThetaHA -4* ThetaHB -4* ThetaQA +2* ThetaQB ) \
    -4*( ImF -ImI )*(hA -vA)*(hB -vB)*np.sin(4* ThetaHA -4* ThetaHB -2* ThetaQA +4* ThetaQB ))
    return prob





def GetTotFlux( counts , params ):
    Uncerts = params [" Uncerts "]
    #(Phi + Phi^T)*( Psi + Psi ^T) = 1
    # measurement order :
    # [H V D A L R] * [H V D A L R] = \
    # H[H V D A L R] V[H V D A L R] D[H V D A L R] A[H V D A L R] L[H V D A L R] R[H V D A L R]
    #(h+v)*(h+v) = hh + hv + vh + vv --> 0 ,1 ,6 ,7
    #(h+v)*(d+a) = hd + ha + vd + va --> 2 ,3 ,8 ,9
    #(h+v)*(l+r) = hl + hr + vl + vr --> 4 ,5 ,10 ,11
    #(d+a)*(h+v) = dh + dv + ah + av --> 12 ,13 ,18 ,19
    #(d+a)*(d+a) = dd + da + ad + aa --> 14 ,15 ,20 ,21
    #(d+a)*(l+r) = dl + dr + al + ar --> 16 ,17 ,22 ,23
    #(l+r)*(h+v) = lh + lv + rh + rv --> 24 ,25 ,30 ,31
    #(l+r)*(d+A) = ld + la + rd + ra --> 26 ,27 ,32 ,33
    #(l+R)*(l+r) = ll + lr + rl + rr --> 28 ,29 ,34 ,35
    ThA = Uncerts [" ThA "]
    ThB = Uncerts [" ThB "]
    TvA = Uncerts [" TvA "]
    TvB = Uncerts [" TvB "]
    RhA = Uncerts [" RhA "]
    RhB = Uncerts [" RhB "]
    RvA = Uncerts [" RvA "]
    RvB = Uncerts [" RvB "]
    MuA = Uncerts [" MuA "]
    NuA = Uncerts [" NuA "]
    MuB = Uncerts [" MuB "]
    NuB = Uncerts [" NuB "]
    # crosstalk and coupling efficiencies muddy our estimate of the total flux
    #( [ muA* ThA muA * TvA ] \ otimes [ muB* ThB muA * TvB ] ) \dot( [hA] \ otimes [hB] )
    #( [ nuA* RhA nuA * RvA ] [ nuB* RhB nuA * RvB ] )( [vA] [VB] )
    crsstlkPBS_A = np.array([[ MuA *ThA , MuA * TvA ],[ NuA *RhA , NuA* RvA]])
    crsstlkPBS_B = np.array([[ MuB *ThB , MuB * TvB ],[ NuB *RhB , NuB* RvB]])
    crsstlkMat_A = np.kron(np.eye(3), crsstlkPBS_A )
    crsstlkMat_B = np.kron(np.eye(3), crsstlkPBS_B )
    crsstlkPBS_AB = np.kron( crsstlkMat_A , crsstlkMat_B )
    # multiply our counts by the pseudo -inverse of crsstlkPBS_AB
    pinvCrsstlkAB = np.linalg.pinv( crsstlkPBS_AB )
    fluxEst = np.dot( pinvCrsstlkAB , counts )
    # to get the total flux , need to sum the 4 relevant projections
    TotFlux = np.zeros(9)
    TotFlux [0] = np.sum( fluxEst [[0 ,1 ,6 ,7]])
    TotFlux [1] = np.sum( fluxEst [[2 ,3 ,8 ,9]])
    TotFlux [2] = np.sum( fluxEst [[4 ,5 ,10 ,11]])
    TotFlux [3] = np.sum( fluxEst [[12 ,13 ,18 ,19]])
    TotFlux [4] = np.sum( fluxEst [[14 ,15 ,20 ,21]])
    TotFlux [5] = np.sum( fluxEst [[16 ,17 ,22 ,23]])
    TotFlux [6] = np.sum( fluxEst [[24 ,25 ,30 ,31]])
    TotFlux [7] = np.sum( fluxEst [[26 ,27 ,32 ,33]])
    TotFlux [8] = np.sum( fluxEst [[28 ,29 ,34 ,35]])
    TotFlux = np.round( TotFlux )
    return TotFlux


# with poisson_model :
#     trace = pm.sample( draws = 1500 , tune = 1000 , cores =4, target_accept = 0.98 )

# ppc = pm.sample_posterior_predictive(trace , model = poisson_model , var_names = [" n_obs "])


def BayesianFit( counts , params ):
    # unpack measurements and parameters
    MmentsA = params [" MmentsA "]
    MmentsB = params [" MmentsB "]
    Uncerts = params [" Uncerts "]
    hA = MmentsA [:,2]
    hB = MmentsB [:,2]
    vA = MmentsA [:,3]
    vB = MmentsB [:,3]
    ThA = Uncerts [" ThA "]
    ThB = Uncerts [" ThB "]
    TvA = Uncerts [" TvA "]
    TvB = Uncerts [" TvB "]
    RhA = Uncerts [" RhA "]
    RhB = Uncerts [" RhB "]
    RvA = Uncerts [" RvA "]
    RvB = Uncerts [" RvB "]
    MuA = Uncerts [" MuA "]
    NuA = Uncerts [" NuA "]
    MuB = Uncerts [" MuB "]
    NuB = Uncerts [" NuB "]
    PBS_std = Uncerts [" PBS_std "]
    MuNu_std = Uncerts [" MuNu_std "]
    Theta_std = Uncerts [" Theta_std "]
    # get least -squares estimate of biphoton input flux using PBS crosstalk matrix
    TotFlux = GetTotFlux( counts , params )
    poisson_model = pm.Model()
    with poisson_model :
        # Gaussian approximation to Poisson distribution
        # Split mean and standard deviation using N(x, sigma ) = x + sigma *N(0,1)
        # 9 flux estimates used for 36 measurements
        stdFlux = pm.Normal(" stdFlux ",np.zeros(9),sigma =1, size = 9)
        fluxDist = pmm.abs_( TotFlux.flatten() + stdFlux *np.sqrt(np.max( TotFlux )))
        # combine 9 flux estimates for 36 measurements appropriately
        temp_a = fluxDist [[0 ,0 ,1 ,1 ,2 ,2]]
        temp_b = fluxDist [[3 ,3 ,4 ,4 ,5 ,5]]
        temp_c = fluxDist [[6 ,6 ,7 ,7 ,8 ,8]]
        flux = at.concatenate([ temp_a , temp_a , temp_b , temp_b , temp_c , temp_c ])
        # priors for density matrix calculation
        t0 = pm.Uniform("t0", lower = -1.0 , upper = 1.0 )
        t1 = pm.Uniform("t1", lower = -1.0 , upper = 1.0 )
        t2 = pm.Uniform("t2", lower = -1.0 , upper = 1.0 )
        t3 = pm.Uniform("t3", lower = -1.0 , upper = 1.0 )
        t4 = pm.Uniform("t4", lower = -1.0 , upper = 1.0 )
        t5 = pm.Uniform("t5", lower = -1.0 , upper = 1.0 )
        t6 = pm.Uniform("t6", lower = -1.0 , upper = 1.0 )
        t7 = pm.Uniform("t7", lower = -1.0 , upper = 1.0 )
        t8 = pm.Uniform("t8", lower = -1.0 , upper = 1.0 )
        t9 = pm.Uniform("t9", lower = -1.0 , upper = 1.0 )
        t10 = pm.Uniform(" t10 ", lower = -1.0 , upper = 1.0)
        t11 = pm.Uniform(" t11 ", lower = -1.0 , upper = 1.0)
        t12 = pm.Uniform(" t12 ", lower = -1.0 , upper = 1.0)
        t13 = pm.Uniform(" t13 ", lower = -1.0 , upper = 1.0)
        t14 = pm.Uniform(" t14 ", lower = -1.0 , upper = 1.0)
        t15 = pm.Uniform(" t15 ", lower = -1.0 , upper = 1.0)
        zWP = pm.Normal("zWP",np.zeros(4),sigma = 1, size =(4))
        ThetaQA = MmentsA [:,1] + Theta_std * zWP [0]
        ThetaHA = MmentsA [:,0] + Theta_std * zWP [1]
        ThetaQB = MmentsB [:,1] + Theta_std * zWP [2]
        ThetaHB = MmentsB [:,0] + Theta_std * zWP [3]
        z1 = pm.Normal("z1",np.zeros(8),sigma = 1, size =(8))
        combPBS_std = np.sqrt( PBS_std **2+ MuNu_std **2)
        ThA_Dist = pmm.abs_( ThA * MuA + combPBS_std *z1[0])
        TvA_Dist = pmm.abs_( TvA * MuA + combPBS_std *z1[1])
        RhA_Dist = pmm.abs_( RhA * NuA + combPBS_std *z1[2])
        RvA_Dist = pmm.abs_( RvA * NuA + combPBS_std *z1[3])
        ThB_Dist = pmm.abs_( ThB * MuB + combPBS_std *z1[4])
        TvB_Dist = pmm.abs_( TvB * MuB + combPBS_std *z1[5])
        RhB_Dist = pmm.abs_( RhB * NuB + combPBS_std *z1[6])
        RvB_Dist = pmm.abs_( RvB * NuB + combPBS_std *z1[7])
        # combine t_i components
        # Deterministic variables have a recorded trance but do not add randomness to model
        tr = pm.Deterministic("tr",pmm.sqr(t0)+ pmm.sqr(t1)+pmm.sqr(t2)+ pmm.sqr(t3)+ pmm .sqr(t4) \
        + pmm.sqr(t5)+pmm.sqr(t6)+ pmm.sqr(t7)+ pmm .sqr(t8)+ pmm.sqr(t9) \
        + pmm.sqr( t10 )+ pmm .sqr(t11)+pmm.sqr( t12 )+ pmm.sqr( t13 )+ pmm.sqr( t14 ) \
        + pmm.sqr( t15 ) )
        A = pm.Deterministic( "A",( pmm.sqr(t0)+pmm.sqr(t1)+ pmm.sqr(t2)+pmm .sqr(t4)+ pmm.sqr(t5) \
        + pmm.sqr(t9)+pmm.sqr( t10 ))/tr )
        B = pm.Deterministic( "B",( pmm.sqr(t3)+pmm.sqr(t6)+ pmm.sqr(t7)+pmm .sqr( t11 )+ pmm.sqr( t12 ))/tr )
        C = pm.Deterministic( "C",( pmm.sqr(t8)+pmm.sqr( t13 )+ pmm.sqr( t14 ))/tr )
        D = pm.Deterministic( "D", pmm.sqr( t15 )/tr )
        ReE = pm.Deterministic( "ReE",( t10 * t12 + t1*t3 + t4*t6 + t5*t7 + t11 *t9)/tr )
        ImE = pm.Deterministic( "ImE",( t10 * t11 + t2*t3 + t5*t6 - t4*t7 - t12 *t9)/tr )
        ReF = pm.Deterministic( "ReF",( t10 * t14 + t4*t8 + t13 *t9)/tr )
        ImF = pm.Deterministic( "ImF",( t10 * t13 + t5*t8 - t14 *t9)/tr )
        ReG = pm.Deterministic( "ReG", t15 *t9/tr )
        ImG = pm.Deterministic( "ImG", t15 * t10 /tr )
        ReH = pm.Deterministic( "ReH",( t11 * t13 + t12 * t14 + t6*t8)/tr )
        ImH = pm.Deterministic( "ImH",( t12 * t13 - t11 * t14 + t7*t8)/tr )
        ReI = pm.Deterministic( "ReI",( t11 * t15 )/tr )
        ImI = pm.Deterministic( "ImI",( t12 * t15 )/tr )
        ReJ = pm.Deterministic( "ReJ",( t13 * t15 )/tr )
        ImJ = pm.Deterministic( "ImJ",( t14 * t15 )/tr )
        # probability of output before PBS crosstalk distorts measurement
        prob = 1./32.*(8*A*hA*hB+8*B*hA*hB+8*C*hA*hB+8*D*hA*hB+8*A*hB*vA \
            +8*B*hB*vA+8*C*hB*vA+8*D*hB*vA+8*A*hA*vB+8*B*hA*vB+8*C*hA*vB+8*D*hA*vB+8*A*vA*vB+8*B*vA*vB \
            +8*C*vA*vB+8*D*vA*vB+4*(A+B-C-D)*(hA -vA)*(hB+vB)* pmm.cos(4* ThetaHA ) \
            +(A-B-C+D+2* ReG +2*ReH)*(hA -vA)*(hB -vB)* pmm.cos(4* ThetaHA -4* ThetaHB ) \
            +4*(A-B+C-D)*(hA+vA)*(hB -vB)* pmm.cos(4* ThetaHB ) \
            +(A-B-C+D-2* ReG -2*ReH)*(hA -vA)*(hB -vB)* pmm.cos(4*( ThetaHA + ThetaHB )) \
            +4*(A+B-C-D)*(hA -vA)*(hB+vB)* pmm.cos(4* ThetaHA -4* ThetaQA ) \
            +(A-B-C+D-2* ReG -2*ReH)*(hA -vA)*(hB -vB)* pmm.cos(4* ThetaHA -4* ThetaHB -4* ThetaQA ) \
            -4*( ImG + ImH )*(hA -vA)*(hB -vB)* pmm.cos(4* ThetaHA -4* ThetaHB -2* ThetaQA ) \
            +4*(ImG+ ImH)*(hA -vA)*(hB -vB)* pmm.cos(4* ThetaHA +4* ThetaHB -2* ThetaQA ) \
            +(A-B-C+D+2* ReG +2*ReH)*(hA -vA)*(hB -vB)* pmm.cos(4*( ThetaHA + ThetaHB -ThetaQA )) \
            +4*(A-B+C-D)*(hA+vA)*(hB -vB)* pmm.cos(4* ThetaHB -4* ThetaQB ) \
            -4*( ImG + ImH )*(hA -vA)*(hB -vB)* pmm.cos(4* ThetaHA +4* ThetaHB -2* ThetaQA -4* ThetaQB ) \
            +4*(ImG -ImH)*(hA -vA)*(hB -vB)* pmm.cos(4* ThetaHA +4* ThetaHB -2* ThetaQB ) \
            -4*( ImG -ImH )*(hA -vA)*(hB -vB)* pmm.cos(4* ThetaHA +4* ThetaHB -4* ThetaQA -2* ThetaQB ) \
            +(A-B-C+D+2* ReG +2*ReH)*(hA -vA)*(hB -vB)* pmm.cos(4*( ThetaHA + ThetaHB -ThetaQB )) \
            +(A-B-C+D-2* ReG -2*ReH)*(hA -vA)*(hB -vB)* pmm.cos(4*( ThetaHA + ThetaHB -ThetaQA -ThetaQB )) \
            +(A-B-C+D-2* ReG -2*ReH)*(hA -vA)*(hB -vB)* pmm.cos(4*( ThetaHA -ThetaHB + ThetaQB )) \
            +(A-B-C+D+2* ReG +2*ReH)*(hA -vA)*(hB -vB)* pmm.cos(4*( ThetaHA -ThetaHB -ThetaQA + ThetaQB )) \
            -4*( ImG -ImH )*(hA -vA)*(hB -vB)* pmm.cos(4* ThetaHA -4* ThetaHB +2* ThetaQB ) \
            +4*(ImG -ImH)*(hA -vA)*(hB -vB)* pmm.cos(4* ThetaHA -4* ThetaHB -4* ThetaQA +2* ThetaQB ) \
            -8*( ReG -ReH )*(hA -vA)*(hB -vB)* pmm.cos(4* ThetaHA -4* ThetaHB -2* ThetaQA +2* ThetaQB ) \
            +4*(ImG+ ImH)*(hA -vA)*(hB -vB)* pmm.cos(4* ThetaHA -4* ThetaHB -2* ThetaQA +4* ThetaQB ) \
            +8*(ReG -ReH)*(hA -vA)*(hB -vB)* pmm.cos(4* ThetaHA +4* ThetaHB -2*( ThetaQA + ThetaQB )) \
            +8*(ReF+ ReI)*(hA -vA)*(hB+vB)* pmm.sin(4* ThetaHA ) \
            -2*( ReE -ReF +ReI -ReJ)*(hA -vA)*(hB -vB)* pmm.sin(4* ThetaHA -4* ThetaHB ) \
            +8*(ReE+ ReJ)*(hA+vA)*(hB -vB)* pmm.sin(4* ThetaHB ) \
            +2*(ReE+ ReF -ReI -ReJ )*(hA -vA)*(hB -vB)*pmm .sin(4*( ThetaHA + ThetaHB )) \
            -8*( ReF + ReI )*(hA -vA)*(hB+vB)* pmm.sin(4* ThetaHA -4* ThetaQA ) \
            -2*( ReE + ReF -ReI -ReJ)*(hA -vA)*(hB -vB)* pmm.sin(4* ThetaHA -4* ThetaHB -4* ThetaQA ) \
            -16*( ImF + ImI )*(hA -vA)*(hB+vB)*pmm.sin(4* ThetaHA -2* ThetaQA ) \
            -4*( ImF -ImI )*(hA -vA)*(hB -vB)* pmm.sin(4* ThetaHA -4* ThetaHB -2* ThetaQA ) \
            -4*( ImF -ImI )*(hA -vA)*(hB -vB)* pmm.sin(4* ThetaHA +4* ThetaHB -2* ThetaQA ) \
            +2*(ReE -ReF+ ReI -ReJ )*(hA -vA)*(hB -vB)*pmm .sin(4*( ThetaHA + ThetaHB -ThetaQA )) \
            -8*( ReE + ReJ )*(hA+vA)*(hB -vB)* pmm.sin(4* ThetaHB -4* ThetaQB ) \
            -4*( ImF -ImI )*(hA -vA)*(hB -vB)* pmm.sin(4* ThetaHA +4* ThetaHB -2* ThetaQA -4* ThetaQB ) \
            -16*( ImE + ImJ )*(hA+vA)*(hB -vB)*pmm.sin(4* ThetaHB -2* ThetaQB ) \
            -4*( ImE -ImJ )*(hA -vA)*(hB -vB)* pmm.sin(4* ThetaHA +4* ThetaHB -2* ThetaQB ) \
            -4*( ImE -ImJ )*(hA -vA)*(hB -vB)* pmm.sin(4* ThetaHA +4* ThetaHB -4* ThetaQA -2* ThetaQB ) \
            -2*( ReE -ReF +ReI -ReJ)*(hA -vA)*(hB -vB)* pmm.sin(4*( ThetaHA + ThetaHB -ThetaQB )) \
            -2*( ReE + ReF -ReI -ReJ)*(hA -vA)*(hB -vB)* pmm.sin(4*( ThetaHA + ThetaHB -ThetaQA -ThetaQB )) \
            +2*(ReE+ ReF -ReI -ReJ )*(hA -vA)*(hB -vB)*pmm .sin(4*( ThetaHA -ThetaHB + ThetaQB )) \
            +2*(ReE -ReF+ ReI -ReJ )*(hA -vA)*(hB -vB)*pmm .sin(4*( ThetaHA -ThetaHB -ThetaQA + ThetaQB )) \
            +4*(ImE -ImJ)*(hA -vA)*(hB -vB)* pmm.sin(4* ThetaHA -4* ThetaHB +2* ThetaQB ) \
            +4*(ImE -ImJ)*(hA -vA)*(hB -vB)* pmm.sin(4* ThetaHA -4* ThetaHB -4* ThetaQA +2* ThetaQB ) \
            -4*( ImF -ImI )*(hA -vA)*(hB -vB)* pmm.sin(4* ThetaHA -4* ThetaHB -2* ThetaQA +4* ThetaQB ))
        # construct joint -space crosstalk matrix
        CrsstlkMatA = at.slinalg.kron(at.eye(3), \
                                          at.stack([ ThA_Dist , TvA_Dist , RhA_Dist , RvA_Dist ]).reshape((2,2)))
        CrsstlkMatB = at.slinalg.kron(at.eye(3), \
                                          at.stack([ ThB_Dist , TvB_Dist , RhB_Dist , RvB_Dist ]).reshape((2,2)))
        CrsstlkMatAB = at.slinalg.kron( CrsstlkMatA , CrsstlkMatB )
        # apply crosstalk to noiseless counts
        NoiselessCounts = prob * flux
        Ncounts = pmm .dot( CrsstlkMatAB , NoiselessCounts.reshape(36 ,1))
        # likelihood distribution for number of observed counts
        n_obs = pm.TruncatedNormal(" n_obs " , mu= Ncounts.flatten(), \
                                     sigma =np.sqrt(np.max( counts )),lower =0, observed = counts.flatten())
    with poisson_model :
        trace = pm.sample( draws = 1500 , tune = 1000 , cores =4, target_accept = 0.98 )
    # posterior predictive checks
    ppc = pm.sample_posterior_predictive(trace , model = poisson_model , var_names = [" n_obs "])
    # plot trace variables and print summary
    az.plot_trace(trace , compact = True ); plt.show()
    print(az.summary(trace , round_to =2))
    return [trace ,ppc , poisson_model ]



# simulate data collection and state reconstruction in main()
def main():
    # set the random -number generating seed to duplicate results
    RANDOM_SEED = 12345
    np.random.seed( RANDOM_SEED )
    # define a two -photon flux
    TestFlux = 1000
    # Maximally -mixed Bell -singlet state
    TrueState = .5*np.array([[0.0 , 0.0 , 0.0 ,0.0 ] ,\
    [0.0 , 1.0 , -1.0 ,0.0 ], \
    [0.0 , -1.0 , 1.0 ,0.0 ], \
    [0.0 , 0.0 , 0.0 ,0.0 ]])
    # density matrix has form
    # A ReE -iImE ReF -iImF ReG -iImG
    # ReE+ iImE B ReH -iImH ReI -iImI
    # ReF+ iImF ReH + iImH C ReJ -imJ
    # ReG+ iImG ReI + iImI ReJ + imJ D
    A = TrueState [0,0]
    B = TrueState [1,1]
    C = TrueState [2,2]
    D = TrueState [3,3]
    ReE = np.real( TrueState [1,0])
    ImE = np.imag( TrueState [1,0])
    ReF = np.real( TrueState [2,0])
    ImF = np.imag( TrueState [2,0])
    ReG = np.real( TrueState [3,0])
    ImG = np.imag( TrueState [3,0])
    ReH = np.real( TrueState [2,1])
    ImH = np.imag( TrueState [2,1])
    ReI = np.real( TrueState [3,1])
    ImI = np.imag( TrueState [3,1])
    ReJ = np.real( TrueState [3,2])
    ImJ = np.imag( TrueState [3,2])
    # waveplate standard deviations / uncertainty
    HWPA_std = 2*np.pi/ 180.
    HWPB_std = 2*np.pi/ 180.
    QWPA_std = 2*np.pi/ 180.
    QWPB_std = 2*np.pi/ 180.
    # fiber coupling efficiencies
    # Mu = PBS transmitted coupling efficiency
    # Nu = PBS reflected coupling efficiency
    MuNu_std = .02
    MuA = .6
    MuB = .7
    NuA = .6
    NuB = .8
    # Include PBS crosstalk
    # values reflect PBS can be lossy
    # assume we can only measure to an accuracy of 2%
    PBS_std = .02
    ThA = .98 # PBS transmission of horizontal polarization for system A
    RhA = .01 # PBS transmission of horizontal polarization for system A
    RvA = .97 # PBS transmission of vertical polarization for system A
    TvA = .01 # PBS transmission of vertical polarization for system A
    ThB = .96 # PBS transmission of horizontal polarization for system B
    RhB = .01 # PBS transmission of horizontal polarization for system B
    RvB = .97 # PBS transmission of vertical polarization for system B
    TvB = .01 # PBS transmission of vertical polarization for system B
    # measurements should follow [HWP , QWP , H, V]
    # where QWP and HWP are fast axis angles(w.r.t.horizontal axis )
    # and H and V are either 0 or 1 depending on the projection
    Mments = np.array([ [0., 0., 1, 0], \
    [0., 0., 0, 1], \
    [np.pi/8, np.pi/4, 1, 0], \
    [np.pi/8, np.pi/4, 0, 1], \
    [0., np.pi/4, 1, 0], \
    [0., np.pi/4, 0, 1] ])
    # Will be generating 36 measurements
    MmentsA = np.kron( Mments ,np.ones([6,1]))
    MmentsB = np.kron(np.ones([6,1]),Mments )
    # assume angles defined with a standard deviation of 2*pi/ 180
    Theta_std = 2*np.pi/ 180
    # systematic deviations in all measurements
    ThetaQA = MmentsA [:,1] + np.random.normal(0, Theta_std )
    ThetaQB = MmentsB [:,1] + np.random.normal(0, Theta_std )
    ThetaHA = MmentsA [:,0] + np.random.normal(0, Theta_std )
    ThetaHB = MmentsB [:,0] + np.random.normal(0, Theta_std )
    hA = MmentsA [:,2]
    hB = MmentsB [:,2]
    vA = MmentsA [:,3]
    vB = MmentsB [:,3]
    # probabilities of measurement output( before sorting with PBS )
    prob = Prob( MmentsA , MmentsB , TrueState )
    # construct crosstalk matrix for system A
    PBS_A_crsstlk = np.zeros([2,2])
    PBS_A_crsstlk [0,0] = MuA * ThA
    PBS_A_crsstlk [0,1] = MuA * TvA
    PBS_A_crsstlk [1,0] = NuA * RhA
    PBS_A_crsstlk [1,1] = NuA * RvA
    # add systematic error in system A's crosstalk components
    PBS_A_crsstlk = np.abs( PBS_A_crsstlk +np.random.normal(0,np.sqrt( PBS_std **2 + MuNu_std **2),(2,2)))
    # probabilites cannot exceed 1
    PBS_A_crsstlk [ PBS_A_crsstlk > 1] = 1
    # construct crosstalk matrix for system B
    PBS_B_crsstlk = np.zeros([2,2])
    PBS_B_crsstlk [0,0] = MuB * ThB
    PBS_B_crsstlk [0,1] = MuB * TvB
    PBS_B_crsstlk [1,0] = NuB * RhB
    PBS_B_crsstlk [1,1] = NuB * RvB
    # add systematic error in system A's crosstalk components
    PBS_B_crsstlk = np.abs( PBS_B_crsstlk +np.random.normal(0,np.sqrt( PBS_std **2 + MuNu_std **2),(2,2)))
    # probabilities cannot exceed 1
    PBS_B_crsstlk [ PBS_B_crsstlk > 1] = 1
    # construct joint -space crosstalk matrix for system AB
    CrsstlkMatA = np.kron(np.eye(3), PBS_A_crsstlk )
    CrsstlkMatB = np.kron(np.eye(3), PBS_B_crsstlk )
    CrsstlkMatAB = np.kron( CrsstlkMatA , CrsstlkMatB )
    # apply crosstalk to flux of a perfect system
    n = np.dot( CrsstlkMatAB , prob * TestFlux )
    # apply Poisson noise
    counts = np.random.poisson(n)
    # group settings and uncertainties in a dictionary
    Uncerts = {" ThA ": ThA ,
    " ThB ": ThB ,
    " TvA ": TvA ,
    " TvB ": TvB ,
    " RhA ": RhA ,
    " RhB ": RhB ,
    " RvA ": RvA ,
    " RvB ": RvB ,
    " MuA ": MuA ,
    " NuA ": NuA ,
    " MuB ": MuB ,
    " NuB ": NuB ,
    " PBS_std ": PBS_std ,
    " MuNu_std ": MuNu_std ,
    " Theta_std ": Theta_std }
    # group measurements and uncertainty into a dictionary settings for cleaner code
    params = {" MmentsA ": MmentsA , " MmentsB ": MmentsB , " Uncerts ": Uncerts }
    # construct state estimate
    print('sab changa si')

    [trace ,ppc , poisson_model ] = BayesianFit( counts , params )
if __name__ == " __main__ ":
    main()


# # traces for two -photon Stokes parameters can be obtained by adding following to model
#S00 = pm.Deterministic( "S00", A + B + C + D ); S01 = pm.Deterministic( " S01", 2*(ReE + ReJ) )
#print(S00)
# S02 = pm.Deterministic( "S02", 2*( ImE + ImJ ) ); S03 = pm.Deterministic( " S03 ", A - B + C - D )
# S10 = pm.Deterministic( "S10", 2*( ReF + ReI ) ); S11 = pm.Deterministic( " S11 ", 2*( ReG + ReH ) )
# S12 = pm.Deterministic( "S12", 2*( ImG - ImH ) ); S13 = pm.Deterministic( " S13 ", 2*( ReF - ReI ) )
# S20 = pm.Deterministic( "S20", 2*( ImF + ImI ) ); S21 = pm.Deterministic( " S21 ", 2*( ImG + ImH ) )
# S22 = pm.Deterministic( "S22", 2*( ReH - ReG ) ); S23 = pm.Deterministic( " S23 ", 2*( ImF - ImI ) )
# S30 = pm.Deterministic( "S30", A + B - C - D ); S31 = pm.Deterministic( " S31 ", 2*(ReE - ReJ) )
# S32 = pm.Deterministic( "S32", 2*( ImE - ImJ ) ); S33 = pm.Deterministic( " S33 ", A - B - C + D )
