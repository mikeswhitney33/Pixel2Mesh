import numpy as np
import cv2
import trimesh
# import shapely
import shapely.geometry
from shapely.geometry import Polygon

def make_rough_model(front_seg, side_seg,contour_method=0):
    """
    Takes a front segmentation & side segmentation of a 3D object (as ndarrays). Currently, y-axis is "up"
    Returns a dict containing the vertices and faces of a rough 3D mesh
    """
    # Available algorithms for point extraction for the front view segmentation
    contour_methods = [cv2.CHAIN_APPROX_NONE, cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_TC89_L1, cv2.CHAIN_APPROX_TC89_KCOS]

    # Define Helper functions to normalize & center points
    normalize_points = lambda p: 2*(p-p.min(axis=0))/np.max(p.max(axis=0)-p.min(axis=0)) - 1
    normalize_depth = lambda p : (2*(p-p.min(axis=0))/(p.max(axis=0)-p.min(axis=0))-1)*p.max()
    ind_convert = lambda p: np.vstack((p[:,0], p[:,1])).T

    # Get the outline contours & hole contours from the front image
    ret, thresh = cv2.threshold(front_seg, 127, 255, 0)                                           #The values here might be wrong if 0/1
    contours, hierarchy,_ = cv2.findContours(thresh, cv2.RETR_CCOMP, contour_methods[contour_method])

    # Sort contours to have largest first, and remove holes with less than 3 points
    sorted_hierarchy = [x.squeeze(1) for x in sorted(hierarchy, key=lambda x: -x.shape[0]) if len(x) > 2]
    
    # Adjust points so (0,0) is the bottom left instead of top left
    contour_outline = ind_convert(sorted_hierarchy[0])
    contour_holes = [ind_convert(sorted_hierarchy[i]) for i in range(1,len(sorted_hierarchy)-1)]

    # Use the side image to compute depth
    ret, thresh = cv2.threshold(side_seg, 127, 255, 0)
    x_len, y_len = thresh.shape
    depth = max(np.array([cv2.countNonZero(thresh[y])/y_len for y in contour_outline[:,1]]))

    
    # Generate the 3d Mesh through extrusion
    poly = Polygon(shell=contour_outline, holes=contour_holes).simplify(1)
    
    # Make sure polygon is valid by applying necessary simplification to rough edges
    simplify_degree = 0
    while not poly.is_valid and simplify_degree <= 5:
        poly = poly.simplify(simplify_degree)
        simplify_degree +=1
    
    # If the polygon is still not valid, we remove the holes and just use the outer contour
    if not poly.is_valid:
        poly = Polygon(shell=contour_outline).simplify(1)
        
        # Make sure polygon is valid by applying necessary simplification to rough edges
        simplify_degree = 0
        while not poly.is_valid and simplify_degree <= 5:
            poly = poly.simplify(simplify_degree)
            simplify_degree +=1
    
    mesh = trimesh.creation.extrude_polygon(poly, depth)

    # Center mesh vertices & depth around origin
    mesh.vertices[:,:2] = normalize_points(mesh.vertices[:,:2])
    mesh.vertices[:,2] = normalize_depth(mesh.vertices[:,2])

    # Make sure that the y coordinate is "up" by adjusting vertices and faces
    mesh.vertices = mesh.vertices[:,np.array([0,2,1])]
    mesh.faces = mesh.faces[:,np.array([0,2,1])]

    # Write the Vertices and faces to the output dictionary representing the obj
    obj_dict={}
    obj_dict['vertices'] = np.array(mesh.vertices)
    obj_dict['faces'] = np.array(mesh.faces)

    return obj_dict