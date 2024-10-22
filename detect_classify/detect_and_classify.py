import sys

sys.path.append('../object_detection')
from segmentation import *
from connected import *
from occupancy_grid import *
sys.path.append('../classification')
from features_extraction import *
from sklearn.externals import joblib
import argparse


def detect_classify(fname, classifier):
    print('Processing {}'.format(fname))
    # finding out the occupancy grid
    occ_grid,point_cloud = occupancy_grid(fname)
    res_grid = np.zeros((occ_grid.shape[0], occ_grid.shape[1]))
    res_grid[occ_grid >= 0.15] = 1
    res_grid[occ_grid < 0.15] = 0
    # obtaining connected components
    conn = Connected(res_grid)
    res = conn.connected_components()
    # selecting components which have more than 10 elements in them
    selected_labels = conn.refining_comps()
    res_z_vals = conn.getting_max_z(selected_labels, occ_grid)
    
    num = 1
    # Applying eigen decomposition to each component
    for i in selected_labels:
        indexes = np.where(res == i)
        cck = np.zeros((2, len(indexes[0])))
        cck_old = np.zeros((2, len(indexes[0])))
        mod_cck = conn.number_in_comps[i]
        if mod_cck >= 100:
            cck_old[0,:] = indexes[0] 
            cck_old[1,:] = indexes[1]
            cck[0,:] = indexes[0] - np.mean(indexes[0])
            cck[1,:] = indexes[1] - np.mean(indexes[1])
            
            cck_trans = cck.transpose()
            # computing co-variance matrix
            cov_mat = np.matmul(cck , cck_trans)/mod_cck
            e_vals, e_vecs = LA.eigh(cov_mat)
            
            new_eigen_vecs = np.column_stack((e_vecs[:,0], e_vecs[:,1]))
        
            # calculating the new coordinates
            new_coords = np.matmul(new_eigen_vecs, cck_old)
            x_min = np.min(new_coords[0,:])
            x_max = np.max(new_coords[0,:])
            y_min = np.min(new_coords[1,:])
            y_max = np.max(new_coords[1,:])
            
            # dimenstions of the box enclosing the segments in the point clouds.
            width = (x_max - x_min)/2
            height = (y_max - y_min)/2
            
            # center of gravity of each component.
            x_bar = np.sum(new_coords[0,:])/mod_cck
            y_bar = np.sum(new_coords[1,:])/mod_cck
        
            z_value = res_z_vals[i]
            
            # Rotating from ego to grid coordinates
            twod_points = np.array([[x_bar-width, y_bar-height],[x_bar+width, y_bar-height],
                                    [x_bar+width, y_bar+height],[x_bar-width, y_bar+height]])
            
            new_points = np.matmul(new_eigen_vecs.transpose(), twod_points.transpose()) 
            points = np.concatenate((new_points.transpose(), np.zeros((4,1))), axis=1)
            points =  np.vstack((points, points))
            points[4:,2] = z_value
        
            # getting point cloud data from the occupancy grid values
            point_x1 =  ((np.min(points[:,0]) - 334) *15/ 100)
            point_x2 = ((np.max(points[:,0]) - 334) * 15/100)
            point_y1 = ((np.min(points[:,1]) - 334) * 15/100)
            point_y2 = ((np.max(points[:,1]) - 334) * 15/100)
            box_3d = point_cloud[(point_cloud[:,0]  >= point_x1) & 
                               (point_cloud[:,0]  <= point_x2) &
                               (point_cloud[:,1]  >= point_y1) &
                               (point_cloud[:,1]  <= point_y2)]
            raw_points = ((points[:4, :2] - 334) * 15)/100
            # refining the point cloud pertaining to a component.
            object_points = scan_conversion(raw_points, box_3d)
            # saving point cloud data related to a component in a txt file.
            # fol_name = int(os.path.basename(os.path.dirname(fname)))
            # f_num = int(os.path.basename(fname)[:-4])
            # bname = '/scratch/ramrao/objects/fol' + str(fol_name) + '_' + str(f_num)+ 'obj' +str(num) + '.txt'
            bname = '/home/ramrao/velodyne/objects/fol1' + '_' + 'obj' +str(num) + '.txt'
            print('Saving to {}'.format(bname))
            np.savetxt(bname, object_points)
            data = feature_extraction(bname)
            classify = joblib.load(classifier)
            res_class =  classify.predict([data])
            class_names = ['non-vehicles', 'vehicles']
            print('resulting class of object num {} is {}'.format(num, class_names[int(res_class)]))
            num = num+1 

    return


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('source' , nargs='+', help='A sequence to be processed')
    parser.add_argument('source' , help='A sequence to be processed')
    parser.add_argument('classifier', help='the classifier to be used for classification (.pkl file)')
    args = parser.parse_args()
    
    detect_classify(args.source, args.classifier)
    return


if __name__ == '__main__':
    main()


