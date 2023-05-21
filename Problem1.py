import numpy as np
import math
import scipy
def calculate_projection_matrix(world_points,image_points):
    A = []
    for i in range(len(world_points)):
        x,y,z = world_points[i]
        u,v= image_points[i]
        a1 = [x,y,z,1,0,0,0,0,-u*x,-u*y,-u*z,-u]
        a2 = [0,0,0,0,x,y,z,1,-v*x,-v*y,-v*z,-v]
        A.append(a1)
        A.append(a2)
    A = np.array(A)
    A_final = np.matmul(A.T,A)
    eig_values,eig_vectors = np.linalg.eig(A_final)
    eig_min_index = np.argmin(eig_values)
    eig_vector_min = np.array(eig_vectors[:,eig_min_index])
    p  = eig_vector_min.reshape(3,4)
    p_final = (1/p[-1,-1])*p
    print("The projection matrix is:")
    print(p_final)    
    # U, s, V = np.linalg.svd(A)
    # H = V[-1].reshape(3,4)
    # print(H[-1,-1])
    return p_final
def reprojection_error(p):
    world_points = [(0,0,0,1),(0,3,0,1),(0,7,0,1),(0,11,0,1),(7,1,0,1),(0,11,7,1),(7,9,0,1),(0,1,7,1)]
    image_points = [(757,213),(758,415),(758,686),(759,966),(1190,172),(329,1041),(1204,850),(340,159)]
    mean = 0
    for i in range(len(world_points)):
        a = np.array(world_points[i])
        calculated_cam_pose = np.matmul(p,a)
        calculated_cam_pose = calculated_cam_pose/calculated_cam_pose[-1]
        reprojection_error = image_points[i] - calculated_cam_pose[:-1]
        mean += np.linalg.norm(reprojection_error)
        print(f"The reprojection error for point {i+1} is ", np.linalg.norm(reprojection_error))
    print("The mean reprojectin error for all the points is",mean/len(world_points))
def decompose_projection_matrix(p):
    p_3_3 = p[:3,:3]  
    p_trans  = p[:,3]
    p_trans  = p_trans.reshape(3,1)
    # p_f = np.flipud(p.T)
    rotation_matrix,intrinsic_matrix = np.linalg.qr(p_3_3)
    intrinsic_matrix = intrinsic_matrix/intrinsic_matrix[-1,-1]
    print("The intrinsic matrix is given as:")
    print(intrinsic_matrix)
    print("The rotation matrix is given as :")
    print(rotation_matrix)
    translation_matrix = np.matmul(np.linalg.inv(intrinsic_matrix),(p_trans))
    print("The translation matrix is given as:")
    print(translation_matrix)
    # r2,q2 = scipy.linalg.rq(p_f)
    # print(q2,"h",r2/r2[-1,-1])
    return
    
    
world_points = [(0,0,0),(0,3,0),(0,7,0),(0,11,0),(7,1,0),(0,11,7),(7,9,0),(0,1,7)]
image_points = [(757,213),(758,415),(758,686),(759,966),(1190,172),(329,1041),(1204,850),(340,159)]
p = calculate_projection_matrix(world_points,image_points)
decompose_projection_matrix(p)
reprojection_error(p)



    
