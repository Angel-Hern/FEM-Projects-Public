#%%
# Initialization
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sympy import symbols, lambdify
from sympy.plotting import plot as symplot
from tabulate import tabulate
import time
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection



NeX = 20
NeY = 20
NeZ = 20

Lx = 0.05 # m
Ly = 0.05 # m
Lz = 0.05 # m

intTemp = 300 # K
k = 35 # W m^-2 K^-1
k_tensor = k * np.eye(3) # W m^-2 K^-1
I0 = 5e7 # W/m^2
a_const = 100 # 1/m
cp = 400 # J/(kg K)
rho = 5500 # kg/m^3
P_abs = lambda x, y, z : (0.02 <= x <= 0.03) * (0.02 <= y <= 0.03) * a_const * I0 * np.exp(-a_const * (Lz - z))
qn = 0 # W/m^2


### Helper Functions ###

def Mesh3D(NeX, NeY, NeZ, Lx, Ly, Lz):
    # Number of Nodes (Nn) in each axis (X,Y,Z)
    NnX = NeX + 1
    NnY = NeY + 1
    NnZ = NeZ + 1

    # Total Number of elements (Ne) in the domain
    Ne = NeX * NeY * NeZ
    # Total Number of nodes (Nn) in the domain
    Nn = NnZ * NnX * NnY

    # Initializatoin of global node coords
    xglobe = np.zeros([Nn, 3])

    hx = Lx / NeX
    hy = Ly / NeY
    hz = Lz / NeZ

    count = 0
    for k in range(NnZ):
        for j in range(NnY):
            for i in range(NnX):
                xglobe[count, :] = np.array([i*hx, j*hy, k*hz])
                count += 1

    # Function that gets node number from (i, j, k)
    # maybe remove the 1 + instead
    f = lambda i, j, k : i + NnX * (j + NnY*k)

    # Initialization
    conn = np.zeros([Ne, 8], dtype=np.int64)
    
    # Keep track of element number
    e = 0
    for k in range(NeZ):
        for j in range(NeY):
            for i in range(NeX):

                Ne_list = [f(i, j, k), f(i+1, j, k),
                           f(i+1, j+1, k), f(i, j+1, k),
                            f(i, j, k+1), f(i+1, j, k+1), 
                            f(i+1, j+1, k+1), f(i, j+1, k+1)]
                conn[e, :] = np.array(Ne_list)
                e += 1


    return xglobe, Ne, Nn, conn


### Provided Functions (translated from MATLAB) ###

def EvalShape2D(pts):
    # pts is expected to be an (N, 2) array with columns [zeta1, zeta2]
    zeta1 = pts[:, 0]
    zeta2 = pts[:, 1]

    # Shape functions
    ShapeFunc2D = np.stack([
        0.25 * (1 - zeta1) * (1 - zeta2),
        0.25 * (1 + zeta1) * (1 - zeta2),
        0.25 * (1 + zeta1) * (1 + zeta2),
        0.25 * (1 - zeta1) * (1 + zeta2)
    ], axis=1)

    # Derivatives w.r.t zeta1
    ShapeDerZeta1 = np.stack([
        -0.25 * (1 - zeta2),
         0.25 * (1 - zeta2),
         0.25 * (1 + zeta2),
        -0.25 * (1 + zeta2)
    ], axis=1)

    # Derivatives w.r.t zeta2
    ShapeDerZeta2 = np.stack([
        -0.25 * (1 - zeta1),
        -0.25 * (1 + zeta1),
         0.25 * (1 + zeta1),
         0.25 * (1 - zeta1)
    ], axis=1)

    return ShapeFunc2D, ShapeDerZeta1, ShapeDerZeta2


def EvalShape3D(pts):
    # pts is an (N, 3) array with columns [zeta1, zeta2, zeta3]
    zeta1 = pts[:, 0]
    zeta2 = pts[:, 1]
    zeta3 = pts[:, 2]

    # Shape functions (phi_i)
    ShapeFunc3D = np.stack([
        1/8 * (1 - zeta1) * (1 - zeta2) * (1 - zeta3),
        1/8 * (1 + zeta1) * (1 - zeta2) * (1 - zeta3),
        1/8 * (1 + zeta1) * (1 + zeta2) * (1 - zeta3),
        1/8 * (1 - zeta1) * (1 + zeta2) * (1 - zeta3),
        1/8 * (1 - zeta1) * (1 - zeta2) * (1 + zeta3),
        1/8 * (1 + zeta1) * (1 - zeta2) * (1 + zeta3),
        1/8 * (1 + zeta1) * (1 + zeta2) * (1 + zeta3),
        1/8 * (1 - zeta1) * (1 + zeta2) * (1 + zeta3),
    ], axis=1)

    # dphi/dzeta1
    ShapeDerZeta1 = np.stack([
        -1/8 * (1 - zeta2) * (1 - zeta3),
         1/8 * (1 - zeta2) * (1 - zeta3),
         1/8 * (1 + zeta2) * (1 - zeta3),
        -1/8 * (1 + zeta2) * (1 - zeta3),
        -1/8 * (1 - zeta2) * (1 + zeta3),
         1/8 * (1 - zeta2) * (1 + zeta3),
         1/8 * (1 + zeta2) * (1 + zeta3),
        -1/8 * (1 + zeta2) * (1 + zeta3),
    ], axis=1)

    # dphi/dzeta2
    ShapeDerZeta2 = np.stack([
        -1/8 * (1 - zeta1) * (1 - zeta3),
        -1/8 * (1 + zeta1) * (1 - zeta3),
         1/8 * (1 + zeta1) * (1 - zeta3),
         1/8 * (1 - zeta1) * (1 - zeta3),
        -1/8 * (1 - zeta1) * (1 + zeta3),
        -1/8 * (1 + zeta1) * (1 + zeta3),
         1/8 * (1 + zeta1) * (1 + zeta3),
         1/8 * (1 - zeta1) * (1 + zeta3),
    ], axis=1)

    # dphi/dzeta3
    ShapeDerZeta3 = np.stack([
        -1/8 * (1 - zeta1) * (1 - zeta2),
        -1/8 * (1 + zeta1) * (1 - zeta2),
        -1/8 * (1 + zeta1) * (1 + zeta2),
        -1/8 * (1 - zeta1) * (1 + zeta2),
         1/8 * (1 - zeta1) * (1 - zeta2),
         1/8 * (1 + zeta1) * (1 - zeta2),
         1/8 * (1 + zeta1) * (1 + zeta2),
         1/8 * (1 - zeta1) * (1 + zeta2),
    ], axis=1)

    return ShapeFunc3D, ShapeDerZeta1, ShapeDerZeta2, ShapeDerZeta3

def Gauss2D(p):
    if p == 2:
        a = 0.577350269189626  # 1/sqrt(3)
        pts2D = np.array([
            [-a, -a],
            [ a, -a],
            [ a,  a],
            [-a,  a]
        ])
        wts2D = np.ones(4)
    
    elif p == 3: 
        a = 0.774596669224148 # sqrt(3/5)
        w1 = 0.555555555555556 
        w2 = 0.888888888888889

        pts2D = np.array([
            [-a, -a],
            [-a,  0],
            [-a,  a],
            [ 0, -a],
            [ 0,  0],
            [ 0,  a],
            [ a, -a],
            [ a,  0],
            [ a,  a]
        ])

        wts2D = np.array([
            w1 * w1,
            w1 * w2,
            w1 * w1,
            w2 * w1,
            w2 * w2,
            w2 * w1,
            w1 * w1,
            w1 * w2,
            w1 * w1
        ])
    else:
        raise ValueError("Only p=2 or p=3 supported.")
    
    return wts2D, pts2D

def Gauss3D(p):
    if p == 2:
        pts_array = np.array([ 0.577350269189626, -0.577350269189626 ])
        wts_array = np.array([ 1.0, 1.0 ])
    elif p == 3:
        pts_array = np.array([ 0.0, 0.774596669224148, -0.774596669224148 ])
        wts_array = np.array([ 0.888888888888889, 0.555555555555556, 0.555555555555556 ])
    else:
        raise ValueError("Only p=2 or p=3 supported.")

    pts3D = []
    wts3D = []

    for i in range(p):
        for j in range(p):
            for k in range(p):
                pts3D.append([pts_array[i], pts_array[j], pts_array[k]])
                wts3D.append(wts_array[i] * wts_array[j] * wts_array[k])

    pts3D = np.array(pts3D)
    wts3D = np.array(wts3D)

    return wts3D, pts3D


# %%

# Number of Gauss Points
p = 2

wts3D, pts3D = Gauss3D(p)
wts2D, pts2D = Gauss2D(p)

ShapeFunc3D, ShapeDer3DZeta1, ShapeDer3DZeta2, ShapeDer3DZeta3 = EvalShape3D(pts3D)
ShapeFunc2D, ShapeDer2DZeta1, ShapeDer2DZeta2, = EvalShape2D(pts2D)

[xglobe, Ne, Nn, conn] = Mesh3D(NeX, NeY, NeZ, Lx, Ly, Lz)


### FEM Solver ###

K = np.zeros([Nn, Nn])
M = np.zeros([Nn, Nn])
R = np.zeros(Nn)
uh = intTemp * np.ones(Nn)
lumped_cap = 1
alt_lumped_cap = 0
iter = 0
t = 0 # s
timeStep = 0.005
tFinal = .2
# tFinal = timeStep
total_time_solve = 0.0

while t < tFinal:
    if iter == 0:
        print("------------------ FEM Solution --------------------")
        print(f"t = {t:.3f} s")
        print(f"Initial Temperature = {max(uh):.2f} K")
        print(f"Time Step Size: {timeStep} s")
        print(f"Lumped Capacitance Assumption is {'True' if lumped_cap else 'False'}")
        print(f"Iteration = {iter}")
        print("--------------------------------------------------\n")

    start = time.time()

    for e in range(Ne):

        conn_ele = conn[e, :]       # Shape: (8,)
        x_ele = xglobe[conn_ele, :] # Shape: (8, 3)


        for q in range(len(pts3D[:,0])):
            # Shape Functions
            Shape3D_q = ShapeFunc3D[q, :]
            x_ele1 = x_ele[:,0]
            x_ele2 = x_ele[:,1]
            x_ele3 = x_ele[:,2]

            x_zeta1 = np.dot(x_ele1, Shape3D_q)
            x_zeta2 = np.dot(x_ele2, Shape3D_q)
            x_zeta3 = np.dot(x_ele3, Shape3D_q)

            # Shape Functions Derivatives w.s.t. zeta1, zeta2, zeta3
            ShapeDer3DZeta1_q = ShapeDer3DZeta1[q,:]
            ShapeDer3DZeta2_q = ShapeDer3DZeta2[q,:]
            ShapeDer3DZeta3_q = ShapeDer3DZeta3[q,:]

            ncv_dphi = np.array([ShapeDer3DZeta1_q, 
                                 ShapeDer3DZeta2_q, 
                                 ShapeDer3DZeta3_q])
            
            # Calculate Deformation Gradient components Fij

            F11 = np.sum(x_ele1 * ShapeDer3DZeta1_q) 
            F12 = np.sum(x_ele1 * ShapeDer3DZeta2_q)
            F13 = np.sum(x_ele1 * ShapeDer3DZeta3_q)

            F21 = np.sum(x_ele2 * ShapeDer3DZeta1_q)
            F22 = np.sum(x_ele2 * ShapeDer3DZeta2_q)
            F23 = np.sum(x_ele2 * ShapeDer3DZeta3_q)

            F31 = np.sum(x_ele3 * ShapeDer3DZeta1_q)
            F32 = np.sum(x_ele3 * ShapeDer3DZeta2_q)
            F33 = np.sum(x_ele3 * ShapeDer3DZeta3_q)

            # F := dx/dzeta
            F = np.array([[F11, F12, F13], 
                          [F21, F22, F23], 
                          [F31, F32, F33]])

            # F^(T) * B = D*phi^(T) ==> B = F^(-T) * D*phi^T
            B = np.linalg.solve(F.T, ncv_dphi)
            
            # det(F)
            J = np.linalg.det(F)

            # Local stiffness matrix ke
            ke_q = wts3D[q] * B.T @ k_tensor @ B * J
            K[np.ix_(conn_ele, conn_ele)] += ke_q
            
            # Local force vector Re
            Re_q = wts3D[q] * Shape3D_q * (P_abs(x_zeta1, x_zeta2, x_zeta3) * J)
            R[conn_ele] += Re_q

            # Mass Matrix 
            me_q = wts3D[q] * rho * (cp * np.outer(Shape3D_q, Shape3D_q) * J)

            if lumped_cap:
            # Sum rows and creates diagonal matrix
                M[np.ix_(conn_ele, conn_ele)] += np.diag(np.sum(me_q, axis=1))
            else:
                M[np.ix_(conn_ele, conn_ele)] += me_q

        # Penatly term
        Penalty = 1000 * max(np.diag(K))

        # Find the nodes associated with a surface
        # array's that are empty are not a outward facing surface

        # Left Surface
        left_idx = (x_ele1 == 0)
        left_surface_nodes = conn_ele[left_idx]

        if left_surface_nodes.size != 0:
            # Correction applied for left surface
            ls_idx = [0, 1, 3, 2]
            left_surface_nodes = left_surface_nodes[ls_idx]
            Surfx_ele = xglobe[left_surface_nodes, :]

            for q in range(len(pts2D[:,0])):
                Shape2D_q = ShapeFunc2D[q, :]
                ShapeDer2DZeta2_q = ShapeDer2DZeta1[q, :]
                ShapeDer2DZeta3_q = ShapeDer2DZeta2[q, :]

                F12 = np.sum(Surfx_ele[:, 0]*ShapeDer2DZeta2_q)
                F13 = np.sum(Surfx_ele[:, 0]*ShapeDer2DZeta3_q)
                F22 = np.sum(Surfx_ele[:, 1]*ShapeDer2DZeta2_q)
                F23 = np.sum(Surfx_ele[:, 1]*ShapeDer2DZeta3_q)
                F32 = np.sum(Surfx_ele[:, 2]*ShapeDer2DZeta2_q)
                F33 = np.sum(Surfx_ele[:, 2]*ShapeDer2DZeta3_q)

                C22 = F12**2 + F22**2 + F32**2
                C33 = F13**2 + F23**2 + F33**2
                C23 = F12*F13 + F22*F23 + F32*F33

                Js = np.sqrt(C22*C33 - C23**2)


                K[np.ix_(left_surface_nodes, left_surface_nodes)] += (
                    Penalty * wts2D[q] * np.outer(Shape2D_q, Shape2D_q) * Js)

                R[left_surface_nodes] += (
                    Penalty * wts2D[q] * Shape2D_q * intTemp * Js)
        
        # Right Surface
        right_idx = (x_ele1 == Lx)
        right_surface_nodes = conn_ele[right_idx]

        if right_surface_nodes.size != 0:
            # No correction needed
            Surfx_ele = xglobe[right_surface_nodes, :]

            for q in range(len(pts2D[:,0])):
                Shape2D_q = ShapeFunc2D[q, :]
                ShapeDer2DZeta2_q = ShapeDer2DZeta1[q, :]
                ShapeDer2DZeta3_q = ShapeDer2DZeta2[q, :]

                F12 = np.sum(Surfx_ele[:, 0]*ShapeDer2DZeta2_q)
                F13 = np.sum(Surfx_ele[:, 0]*ShapeDer2DZeta3_q)
                F22 = np.sum(Surfx_ele[:, 1]*ShapeDer2DZeta2_q)
                F23 = np.sum(Surfx_ele[:, 1]*ShapeDer2DZeta3_q)
                F32 = np.sum(Surfx_ele[:, 2]*ShapeDer2DZeta2_q)
                F33 = np.sum(Surfx_ele[:, 2]*ShapeDer2DZeta3_q)

                C22 = F12**2 + F22**2 + F32**2
                C33 = F13**2 + F23**2 + F33**2
                C23 = F12*F13 + F22*F23 + F32*F33

                Js = np.sqrt(C22*C33 - C23**2)

                
                K[np.ix_(right_surface_nodes, right_surface_nodes)] += (
                    Penalty * wts2D[q] * np.outer(Shape2D_q, Shape2D_q) * Js)

                R[right_surface_nodes] += (
                    Penalty * wts2D[q] * 
                    Shape2D_q * intTemp * Js)

        # Front Surface
        front_idx = (x_ele2 == 0)
        front_surface_nodes = conn_ele[front_idx]

        if front_surface_nodes.size != 0:
            fs_idx = [0, 1, 3, 2]
            front_surface_nodes = front_surface_nodes[fs_idx]
            Surfx_ele = xglobe[front_surface_nodes, :]

            for q in range(len(pts2D[:,0])):
                Shape2D_q = ShapeFunc2D[q, :]
                ShapeDer2DZeta1_q = ShapeDer2DZeta1[q, :]
                ShapeDer2DZeta3_q = ShapeDer2DZeta1[q, :]

                F11 = np.sum(Surfx_ele[:, 0]*ShapeDer2DZeta1_q)
                F13 = np.sum(Surfx_ele[:, 0]*ShapeDer2DZeta3_q)
                F21 = np.sum(Surfx_ele[:, 1]*ShapeDer2DZeta1_q)
                F23 = np.sum(Surfx_ele[:, 1]*ShapeDer2DZeta3_q)
                F31 = np.sum(Surfx_ele[:, 2]*ShapeDer2DZeta1_q)
                F33 = np.sum(Surfx_ele[:, 2]*ShapeDer2DZeta3_q)

                C11 = F12**2 + F21**2 + F31**2
                C13 = F12*F13 + F21*F23 + F31*F33
                C33 = F13**2 + F23**2 + F33**2

                Js = np.sqrt(C11*C33 - C13**2)

                K[np.ix_(front_surface_nodes, front_surface_nodes)] += (
                    Penalty * wts2D[q] * np.outer(Shape2D_q, Shape2D_q) * Js)

                R[front_surface_nodes] += (
                    Penalty * wts2D[q] * Shape2D_q * intTemp * Js)

        # Back Surface
        back_idx = (x_ele2 == Ly)
        back_surface_nodes = conn_ele[back_idx]

        if back_surface_nodes.size != 0:
            bs_idx = [1, 0, 2, 3]
            back_surface_nodes = back_surface_nodes[bs_idx]
            Surfx_ele = xglobe[back_surface_nodes, :]

            for q in range(len(pts2D[:,0])):
                Shape2D_q = ShapeFunc2D[q, :]
                ShapeDer2DZeta1_q = ShapeDer2DZeta1[q, :]
                ShapeDer2DZeta3_q = ShapeDer2DZeta2[q, :]

                F11 = np.sum(Surfx_ele[:, 0]*ShapeDer2DZeta1_q)
                F13 = np.sum(Surfx_ele[:, 0]*ShapeDer2DZeta3_q)
                F21 = np.sum(Surfx_ele[:, 1]*ShapeDer2DZeta1_q)
                F23 = np.sum(Surfx_ele[:, 1]*ShapeDer2DZeta3_q)
                F31 = np.sum(Surfx_ele[:, 2]*ShapeDer2DZeta1_q)
                F33 = np.sum(Surfx_ele[:, 2]*ShapeDer2DZeta3_q)

                C11 = F12**2 + F21**2 + F31**2
                C13 = F12*F13 + F21*F23 + F31*F33
                C33 = F13**2 + F23**2 + F33**2

                Js = np.sqrt(C11*C33 - C13**2)

                K[np.ix_(back_surface_nodes, back_surface_nodes)] += (
                    Penalty * wts2D[q] * np.outer(Shape2D_q, Shape2D_q) * Js)

                R[back_surface_nodes] += (
                    Penalty * wts2D[q] 
                    * Shape2D_q * intTemp * Js)

        # For Neumann Boundary Condition
        # Not necessary because qn = 0, but left in case qn != 0

        # Bottom Surface
        bottom_idx = (x_ele3 == 0)
        bottom_surface_nodes = conn_ele[bottom_idx]

        if bottom_surface_nodes.size != 0:
                # No correction needed
                Surfx_ele = xglobe[bottom_surface_nodes, :]
                # total_val = 0.0
                for q in range(len(pts2D[:,0])):
                    Shape2D_q = ShapeFunc2D[q, :]
                    ShapeDer2DZeta1_q = ShapeDer2DZeta1[q, :]
                    ShapeDer2DZeta2_q = ShapeDer2DZeta2[q, :]

                    F11 = np.sum(Surfx_ele[:,0]*ShapeDer2DZeta1_q)
                    F12 = np.sum(Surfx_ele[:,0]*ShapeDer2DZeta2_q)
                    F21 = np.sum(Surfx_ele[:,1]*ShapeDer2DZeta1_q)
                    F22 = np.sum(Surfx_ele[:,1]*ShapeDer2DZeta2_q)
                    F31 = np.sum(Surfx_ele[:,2]*ShapeDer2DZeta1_q)
                    F32 = np.sum(Surfx_ele[:,2]*ShapeDer2DZeta2_q)

                    C11 = F11**2 + F21**2 + F31**2
                    C22 = F12**2 + F22**2 + F32**2
                    C12 = F11*F12 + F21*F22 + F31*F32

                    Js = np.sqrt(C11*C22 - C12**2)

                    # test_val = wts2D[q] * Shape2D_q * Js
                    # total_val += test_val
                    R[bottom_surface_nodes] += (
                        wts2D[q] * Shape2D_q * qn * Js)


        # Top Surface
        top_idx = (x_ele3 == Lz)
        top_surface_nodes = conn_ele[top_idx]

        if top_surface_nodes.size != 0:
            # No correction needed
            Surfx_ele = xglobe[top_surface_nodes, :]

            for q in range(len(pts2D[:,0])):
                Shape2D_q = ShapeFunc2D[q, :]
                ShapeDer2DZeta1_q = ShapeDer2DZeta1[q, :]
                ShapeDer2DZeta2_q = ShapeDer2DZeta2[q, :]

                F11 = np.sum(Surfx_ele[:,0]*ShapeDer2DZeta1_q)
                F12 = np.sum(Surfx_ele[:,0]*ShapeDer2DZeta2_q)
                F21 = np.sum(Surfx_ele[:,1]*ShapeDer2DZeta1_q)
                F22 = np.sum(Surfx_ele[:,1]*ShapeDer2DZeta2_q)
                F31 = np.sum(Surfx_ele[:,2]*ShapeDer2DZeta1_q)
                F32 = np.sum(Surfx_ele[:,2]*ShapeDer2DZeta2_q)


                C11 = F11**2 + F21**2 + F31**2
                C22 = F12**2 + F22**2 + F32**2
                C12 = F11*F12 + F21*F22 + F31*F32

                Js = np.sqrt(C11*C22 - C12**2)

                R[top_surface_nodes] += (wts2D[q] * Shape2D_q * qn * Js)

    # End of looping through elements

    # Solves for current tempature distrubtion at time t
    # val = M^-1 * ((-K*uh) + R)
    # theta^(L+1) = uh + val*timeStep

    # Use alternative lumped captiance matrix
    if alt_lumped_cap:
        M_scalar = np.sqrt(np.sum(M**2))
        M = M_scalar * np.eye(M.shape[0])

    uhNew = uh + np.linalg.solve(M, np.dot(-K, uh) + R) * timeStep
    t = t + timeStep
    iter += 1
    uh = uhNew
    end = time.time()
    total_time_solve += (end - start)

    print("------------------ FEM Solution --------------------")
    print(f"t = {t:.3f} s")
    print(f"Current Max Temperature = {max(uh):.2f} K")
    print(f"Lumped Capacitance: {'True' if lumped_cap else 'False'}")
    print(f"Time Step Size: {timeStep} s")
    print(f"Iteration = {iter}")
    print(f"Time to Solve = {end - start:.2f} s")
    print("--------------------------------------------------\n")

print("------------------ FEM Solution --------------------")
print(f"Final Time: t = {t:.3f} s")
print(f"Final Max Temperature = {max(uh):.2f} K")
print(f"Given Lumped Capacitance Assumption is {'True' if lumped_cap else 'False'}")
print(f"Total number of Iterations = {iter}")
print(f"Total Time to Solve = {total_time_solve:.2f} s")
print("--------------------------------------------------\n")


# %%

# Slicing function to put into a table

def find_u_at_z_slice(nodes, elements, u, z_slice, xy_values, tol=1e-3):

    values = []

    for xy in xy_values:
        x, y = xy

        # Find all nodes that match (x, y, z_slice) within tolerance
        matches = np.where(
            (np.abs(nodes[:, 0] - x) < tol) &
            (np.abs(nodes[:, 1] - y) < tol) &
            (np.abs(nodes[:, 2] - z_slice) < tol)
        )[0]

        if matches.size > 0:
            for idx in matches:
                values.append(u[idx])
        else:
            values.append(np.nan)  # or some flag for "no match"

    return np.array(values)
        

def plot_hexahedral_mesh_with_colors(nodes, elements, u):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a normalization based on the full range of u
    norm = plt.Normalize(vmin=np.min(u), vmax=np.max(u))
    cmap = cm.viridis

    # Loop through all elements
    for element in elements:
        element_nodes = nodes[element]
        element_solution = u[element]
        draw_hexahedron_with_colors(ax, element_nodes, element_solution, cmap, norm)

    ax.set_box_aspect([1,1,1])  # Equal aspect ratio

    # Set axis labels
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(u)

    # Manually add colorbar in a new axis to the right
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
    fig.colorbar(mappable, cax=cbar_ax, label='Temperature (K)')

    plt.subplots_adjust(right=0.8)  # Leave space for colorbar
    ax.set_title('FEM Solution', pad=20)
    plt.show()

def draw_hexahedron_with_colors(ax, nodes, values, cmap, norm):
    # Define faces of the hexahedron
    faces = [
        [0, 1, 2, 3], [4, 5, 6, 7],  # bottom, top
        [0, 1, 5, 4], [1, 2, 6, 5],
        [2, 3, 7, 6], [3, 0, 4, 7]
    ]

    for face in faces:
        face_vertices = nodes[face]
        face_values = values[face]
        avg_value = np.mean(face_values)
        face_color = cmap(norm(avg_value))

        poly = Poly3DCollection([face_vertices], facecolors=[face_color], edgecolors='k', linewidths=0.5)
        ax.add_collection3d(poly)

def plot_2d_quad_slice(nodes, elements, u, z_slice, tol=1e-3):

    quads = []
    values = []

    for element in elements:
        # Check both bottom and top faces
        for face in ([0, 1, 2, 3], [4, 5, 6, 7]):
            face_nodes = nodes[element[face]]
            face_z = face_nodes[:, 2]

            if np.all(np.abs(face_z - z_slice) < tol):
                xy_coords = face_nodes[:, :2]
                quads.append(Polygon(xy_coords, closed=True))
                face_values = u[element[face]]
                values.append(np.mean(face_values))

    if not quads:
        print(f"No quadrilateral faces found at z = {z_slice}")
        return

    fig, ax = plt.subplots()
    p = PatchCollection(quads, cmap='viridis', edgecolor='k', linewidths=0.5)
    p.set_array(np.array(values))
    ax.add_collection(p)
    fig.colorbar(p, ax=ax, label='Temperature (K)')

    # scaling
    ax.set_xlim(np.min(nodes[:, 0]), np.max(nodes[:, 0]))
    ax.set_ylim(np.min(nodes[:, 1]), np.max(nodes[:, 1]))

    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'2D Cross Sectional Slice at z = {z_slice}')
    plt.show()

xy_values = np.array([[0.0250, 0.0275, 0.0300, 0.0400],
                      [0.0250, 0.0275, 0.0300, 0.0400]])

xy_pairs = xy_values.T
z_values = np.array([Lz, 0.5*Lz, 0])

num_xy = xy_pairs.shape[0]
num_z = len(z_values)

table = []  # (x, y, z, u)

for z in z_values:
    u_vals = find_u_at_z_slice(xglobe, conn, uh, z, xy_pairs, tol=1e-3)  # returns array of length N
    for i, (x, y) in enumerate(xy_pairs):
        u = u_vals[i] if i < len(u_vals) else np.nan
        table.append([x, y, z, u])


df = pd.DataFrame(table, columns=["x", "y", "z", "Temp"])

# Display the table
print(f"Temperature Field Table with time step size {timeStep} \n")
print(f"and Lumped Capacitance Assumption is {'True' if lumped_cap else 'False'}\n")
print(df.to_string(index=False))


# 3D FEM Plot 
plot_hexahedral_mesh_with_colors(xglobe, conn, uh)

# 2D Cross sectional views:
# at the top (z=z_top)
plot_2d_quad_slice(xglobe, conn, uh, Lz, tol=1e-3)
# at the middle (z=1/2 *z_top)
plot_2d_quad_slice(xglobe, conn, uh, Lz*.5, tol=1e-3)
# at the bottom (z = 0)
plot_2d_quad_slice(xglobe, conn, uh, Lz*0, tol=1e-3)