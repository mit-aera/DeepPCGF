import time
import numpy as np
import copy
#import open3d as o3d
# from open3d.open3d.geometry import estimate_normals

def make_open3d_point_cloud(xyz, color=None, visualize=False):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(xyz)
  if color is not None:
    pcd.colors = o3d.utility.Vector3dVector(color)
  if visualize:
    o3d.visualization.draw_geometries([pcd])
  return pcd

def vis_open3d_two_pcds(xyz1, xyz2, color=None):
    color1 = np.zeros_like(xyz1)+ np.array([[0.5, 0, 0]])
    color2 = np.zeros_like(xyz1)+ np.array([[0, 0.5, 0]])
    pcd1 = make_open3d_point_cloud(xyz1, color1)
    pcd2 = make_open3d_point_cloud(xyz2, color2)
    
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[-2, -2, -2])

    o3d.visualization.draw_geometries([pcd1, pcd2, mesh_frame])

def vis_open3d_three_pcds(xyz1, xyz2, xyz3, color=None):
    color1 = np.zeros_like(xyz1)+ np.array([[0.5, 0, 0]])
    color2 = np.zeros_like(xyz2)+ np.array([[0, 0.5, 0]])
    color3 = np.zeros_like(xyz3)+ np.array([[0, 0.0, 0.5]])
    pcd1 = make_open3d_point_cloud(xyz1, color1)
    pcd2 = make_open3d_point_cloud(xyz2, color2)
    pcd3 = make_open3d_point_cloud(xyz3, color3)
    
    o3d.visualization.draw_geometries([pcd1, pcd2, pcd3])

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def icp_refine(R, t, model_points, pts_sensor_cam):
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(model_points)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(pts_sensor_cam)

    threshold = 0.005#0.010#0.025
    temp = np.concatenate((R, t), axis=1)
   
    trans_init = np.concatenate((temp, np.array([[0.0, 0.0, 0.0, 1.0]])))

    # draw_registration_result(source, target, trans_init)
    evaluation = o3d.registration.evaluate_registration(source, target,
                                            threshold, trans_init)
    # print("Initial alignment")
    # print(evaluation)

    reg_p2p = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint(),
	o3d.registration.ICPConvergenceCriteria(max_iteration = 100))
    #source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    #    radius=0.01, max_nn=30))
    #target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    #    radius=0.01, max_nn=30))
    #reg_p2p = o3d.registration.registration_icp(
    #    source, target, threshold, trans_init,
    #    o3d.registration.TransformationEstimationPointToPlane())
    R = reg_p2p.transformation[:3, :3]
    t = reg_p2p.transformation[:3, 3][:, None]
    
    return R, t
    # print("Apply point-to-point ICP")
    # print(reg_p2p)
    # draw_registration_result(source, target, reg_p2p.transformation)
def create_fpfh_feature(pcd, voxel_size):
    estimate_normals(pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_size, max_nn=50))
    fpfh = o3d.registration.compute_fpfh_feature(pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=5*voxel_size, max_nn=50))
    return fpfh

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold,
            ))
    return result

def fgr(pts_model, pts_sensor, voxel_size):
    pcd_sensor = o3d.PointCloud()
    pcd_sensor.points = o3d.Vector3dVector(pts_sensor)
    pcd_model = o3d.PointCloud()
    pcd_model.points = o3d.Vector3dVector(pts_model)

    fpfh_sensor = create_fpfh_feature(pcd_sensor, voxel_size)
    fpfh_model = create_fpfh_feature(pcd_model, voxel_size)
    # draw_registration_result(source_down, target_down,
    #                          result_ransac.transformation)
    
    start = time.time()
    result_fast = execute_fast_global_registration(pcd_model, pcd_sensor,
                                                   fpfh_model, fpfh_sensor,
                                                   voxel_size)
    # print("Fast global registration took %.3f sec.\n" % (time.time() - start))
    draw_registration_result(pcd_model, pcd_sensor,
                             result_fast.transformation)
    return result_fast.transformation

def ransac_global_registration(pts_model, pts_sensor, voxel_size):
    distance_threshold = 0.4*voxel_size
    pcd_sensor = o3d.geometry.PointCloud()
    pcd_sensor.points = o3d.utility.Vector3dVector(pts_sensor)
    pcd_model = o3d.geometry.PointCloud()
    pcd_model.points = o3d.utility.Vector3dVector(pts_model)

    fpfh_sensor = create_fpfh_feature(pcd_sensor, voxel_size)
    fpfh_model = create_fpfh_feature(pcd_model, voxel_size)

    result = o3d.registration.registration_ransac_based_on_feature_matching(
        pcd_model, pcd_sensor, fpfh_model, fpfh_sensor, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    draw_registration_result(pcd_model, pcd_sensor,
                             result.transformation)
    return result.transformation

def ransac_global_correspondence(pts_model, pts_sensor,
     max_correspondence_distance=0.0045,#0.02, # 0.04 for can
     max_iteration=1000,
     max_validation=1000):

    pcd_sensor = o3d.geometry.PointCloud()
    pcd_sensor.points = o3d.utility.Vector3dVector(pts_sensor)
    pcd_model = o3d.geometry.PointCloud()
    pcd_model.points = o3d.utility.Vector3dVector(pts_model)

    corr = np.zeros((len(pts_model), 2))
    corr[:, 0] = np.arange(len(pts_model))
    corr[:, 1] = np.arange(len(pts_model))
    corr = o3d.utility.Vector2iVector(corr)

    criteria = o3d.registration.RANSACConvergenceCriteria(max_iteration, max_validation)
    result = o3d.registration.registration_ransac_based_on_correspondence(
        pcd_model, pcd_sensor, corr, max_correspondence_distance, 
        ransac_n=6, criteria=criteria) # ransac_n: default value 6

    #result = o3d.registration.registration_ransac_based_on_correspondence(
    #    pcd_model, pcd_sensor, corr, max_correspondence_distance, 
    #    ransac_n=3, criteria=criteria)

    return result.transformation