import nibabel as nib
import numpy as np
import os
import hcp_utils as hcp
import pandas as pd
from scipy import stats
import matplotlib
#.......................................................................................


def struct_to_bm(cifti_struct, cifti):
    brain_model = [x for x in cifti.header.get_index_map(1).brain_models if x.brain_structure==cifti_struct]
    del cifti
    return brain_model[0]
#.......................................................................................


def struct_info(cifti_struct, cifti):
    brain_model = [x for x in cifti.header.get_index_map(1).brain_models if x.brain_structure==cifti_struct]
    offset = brain_model[0].index_offset
    count = brain_model[0].index_count
    vertex_indices = np.array(brain_model[0].vertex_indices, dtype='int32')
    
    del cifti
    return offset,count,vertex_indices
#.......................................................................................


def label_df(dlabel_file):
    '''
    Generate a Pandas DataFrame with labels and color information
        dlabel_file: reference cifti2.dlabel file
    '''
    parc_dict = dlabel_file.header.get_axis(0).get_element(0)[1]
    label_tbl = pd.DataFrame([[parc, parc_dict[parc][0], parc_dict[parc][1]] for parc in parc_dict], columns=['label', 'name', 'rgba'])
    
    del dlabel_file
    return label_tbl
#.......................................................................................


def agg_labels(cifti, labels, func='mean'):
    '''
    Aggregate timeseries or scalars based on a specified parcellation.
    This function uses the groupby and agg methods of Pandas datarames to
    aggregate data using a specified function.
    
    cifti: a cifti dtseries or dscalar file
    labels: cifti dlabel file indicating parcels
    func (optional): function to apply to vertices within a parcel (see pandas.DataFrame.agg)
    
    Returns a pandas.DataFrame with a column for each parcel
    '''
    
    
    
    label_tbl = label_df(labels)
    
    # Left brain structure
    dt_os_L, dt_n_L, dt_vv_L = struct_info("CIFTI_STRUCTURE_CORTEX_LEFT", cifti)
    lbl_os_L, lbl_n_L, lbl_vv_L = struct_info("CIFTI_STRUCTURE_CORTEX_LEFT", labels)

    # Right brain structure
    dt_os_R, dt_n_R, dt_vv_R = struct_info("CIFTI_STRUCTURE_CORTEX_RIGHT", cifti)
    lbl_os_R, lbl_n_R, lbl_vv_R = struct_info("CIFTI_STRUCTURE_CORTEX_RIGHT", labels)

    
    lbls = np.hstack([labels.get_fdata()[:,lbl_os_L:lbl_os_L+lbl_n_L],
                      labels.get_fdata()[:,lbl_os_R:lbl_os_R+lbl_n_R]])
    shared_vv_lbl = np.hstack([np.isin(lbl_vv_L, dt_vv_L), np.isin(lbl_vv_R, dt_vv_R)])
    shared_vv_dt = np.hstack([np.isin(dt_vv_L, lbl_vv_L, ), np.isin(dt_vv_R, lbl_vv_R)])
    
    lbls = lbls[:, shared_vv_lbl]
    scals = np.hstack([cifti.get_fdata()[:,lbl_os_L:lbl_os_L+lbl_n_L],
                       cifti.get_fdata()[:,lbl_os_R:lbl_os_R+lbl_n_R]])[:, shared_vv_dt]
    
    
    # merge
    print(lbls)
    df = np.vstack([lbls, scals]).T
    names = np.hstack(['label', cifti.header.get_axis(0).name])
    if len(names)!=df.shape[1]:
        names = [f"Scalar {i}" for i in range(1,df.shape[1])]
        names = np.hstack(['label', names])
    df = pd.DataFrame(df, columns=names).groupby('label').agg(func)
        
    return df
#.......................................................................................


def agg_networks(cifti, networks, func='mean', by_hemisphere=False, label_tbl=False):
    '''
    Aggregate timeseries or scalars based on Yeo et al.'s networks using
    the Shaefer atlas ciftis.
    (https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/HCP/)
    This function uses the groupby and agg methods of Pandas datarames to
    aggregate data using a specified function.
    
    cifti: a cifti dtseries or dscalar file
    labels: cifti dlabel file indicating parcels
    func (optional): function to apply to vertices within a parcel (see pandas.DataFrame.agg)
    by_hemisphere (optional): aggregate network regions separately by hemisphere
    label_tbl (optional): return a pandas dataframe containing regional and network labels, and color information
    
    Returns a pandas.DataFrame with a column for each parcel
    '''
    
    # Left brain structure
    dt_os_L, dt_n_L, dt_vv_L = struct_info("CIFTI_STRUCTURE_CORTEX_LEFT", cifti)
    lbl_os_L, lbl_n_L, lbl_vv_L = struct_info("CIFTI_STRUCTURE_CORTEX_LEFT", networks)
    # Right brain structure
    dt_os_R, dt_n_R, dt_vv_R = struct_info("CIFTI_STRUCTURE_CORTEX_RIGHT", cifti)
    lbl_os_R, lbl_n_R, lbl_vv_R = struct_info("CIFTI_STRUCTURE_CORTEX_RIGHT", networks)
    # mask for shared vertices
    shared_vv = np.hstack([np.isin(lbl_vv_L, dt_vv_L), np.isin(lbl_vv_R, dt_vv_R)])
    
    # replace region labels with network labels
    labels = np.hstack([networks.get_fdata()[:,lbl_os_L:lbl_os_L+lbl_n_L],
                        networks.get_fdata()[:,lbl_os_R:lbl_os_R+lbl_n_R]])
    labels = labels[:, shared_vv]
    labels_tbl = label_df(networks)
    
    
    if by_hemisphere:
        labels_tbl.loc[1:,'name'] = labels_tbl.loc[1:, 'name'].apply(lambda x: '_'.join(x.split('_')[1:3]))
    else:
        labels_tbl.loc[1:,'name']= labels_tbl.loc[1:,'name'].apply(lambda x: x.split('_')[2])
    
    labels_tbl.set_index('name', inplace=True)
    labels_tbl['network'] = labels_tbl['label'].groupby(level='name').transform('min')
    labels_tbl['network'] = stats.rankdata(labels_tbl['network'], method='dense') - 1
    
    for i in labels_tbl.network.unique():
        labels[np.isin(labels, labels_tbl.loc[labels_tbl.network==i, 'label'])] = i

    
    # create dataframe, group by network, and aggregate vertex values
    scals = np.hstack([cifti.get_fdata()[:,lbl_os_L:lbl_os_L+lbl_n_L],
                       cifti.get_fdata()[:,lbl_os_R:lbl_os_R+lbl_n_R]])    
    df = np.vstack([labels, scals]).T
    names = np.hstack(['network', cifti.header.get_axis(0).name])

    df = pd.DataFrame(df, columns=names).groupby('network').agg(func)        
    if label_df:
        return df, labels_tbl
    else:
        return df
#.......................................................................................


def label_colors(dlabel_file):
    labels_tbl = label_df(dlabel_file)
    
    colors = []
    for lbl in range(len(labels_tbl.label.unique())):
        c = labels_tbl.rgba[labels_tbl.label==lbl].values[0]
        colors.append(list(c))
    
    return colors
#.......................................................................................


def label_to_vtx(label, dlabel_file, brain_structure):
    '''
    Get the vertex index of nodes with a specified label
        label: label of the parcel of which you need the nodes (must be in the labels of the dlabel file)
        dlabel_file: reference cifti2.dlabel file
        brain_structure: cifti2 brain structure
    '''
    brain_model = [x for x in dlabel_file.header.get_index_map(1).brain_models if x.brain_structure==brain_structure][0]
    offset = brain_model.index_offset
    count = brain_model.index_count
    vertices = brain_model.vertex_indices[0:]
    vertices = np.asarray(vertices)

    label_map = pd.Series(dlabel_file.get_fdata().squeeze()[offset:offset+count])
    label_lst = pd.DataFrame(dlabel_file.header.get_axis(0).get_element(0)[1].values(),columns=('lab','col')).reset_index()
    label_tmp = label_lst[label_lst['lab'].isin(pd.Series(label))]['index'].values
    label_vtx = np.array(vertices[label_map.isin(label_tmp)], dtype='int32')

    del dlabel_file
    return label_map.isin(label_tmp).to_list(), label_vtx
#.......................................................................................


def struct_to_vtx(cifti_structs, cifti):
    '''
    Extract the vertex indices included in one or more structures
        cifti_structs: list of structures (must be in the BrainModelAxis of the cifti file)
        path_to_cifti: path to the cifti file
    '''
    cifti_structs = ([cifti_structs] if type(cifti_structs)==str else cifti_structs)
    all_structs = np.array(list(cifti.header.get_axis(1).iter_structures()), dtype=object)
    struct_vtx = []    
    for struct in cifti_structs:
        if struct not in all_structs[:,0]:
            print(f"'{struct}' was not fount in the BrainModelAxis of this cifti file")
            continue
        vtx = all_structs[all_structs[:,0]==struct,2][0].vertex
        struct_vtx.append(np.array(vtx))
    del cifti
    return struct_vtx
#.......................................................................................


def vtx_to_surf(indices, surface, index_dict=False):
    vertices = surface.darrays[0].data
    vertices = np.array(vertices[indices], dtype='float64')
    
    # the following paragraph is adapted from https://github.com/NeuroanatomyAndConnectivity/surfdist/blob/master/surfdist/utils.py
    triangles = surface.darrays[1].data
    triangles_old = triangles[np.all(np.isin(triangles, indices),axis=1)]    
    new_indices = np.digitize(triangles_old.ravel(), indices, right=True)
    triangles = np.array(np.arange(len(indices))[new_indices].reshape(triangles_old.shape), dtype=np.int32)
    
    if index_dict:
        index_dict_out = dict(zip(indices, np.argsort(indices)))
        return vertices, triangles, index_dict_out
    else:
        return vertices, triangles
    del surface
#.......................................................................................


def vtx_to_neighbors(vtx, surface):
    trigs = surface.darrays[1].data
    try:
        neighbors = np.unique(np.concatenate(trigs[np.any(trigs==vtx, axis=1)]))
        return neighbors

    except:
        print(f'Vertex {vtx} not in surface')
        
        
        
def vtx_to_roi(vtx, roi_size, dscalar, surface, structure, matrix_row=0, ascending=True):    
    matrix = dscalar.get_fdata()
    os, n, vv = struct_info(structure, dscalar)
    scalar_hemi = matrix[matrix_row, os:os+n]
    
    vtx_i = vtx
    roi = [vtx_i]
    while len(roi) < roi_size:
        # neighbors = vtx_to_neighbors(vtx_i, surface)
        # neighbors = neighbors[~np.isin(neighbors, roi)]
        neighbors = np.array([], dtype='int32')
        step_back = -1
        while neighbors.size == 0:
            vtx_i = roi[step_back]
            neighbors = vtx_to_neighbors(vtx_i, surface)
            neighbors = neighbors[~np.isin(neighbors, roi)]
            step_back -= 1
            
        nbr_scalar = [scalar_hemi[vv==nbr] for nbr in neighbors]
        vtx_i = (neighbors[np.argmax(nbr_scalar)] if ascending else neighbors[np.argmin(nbr_scalar)])
        roi.append(vtx_i)      

    return np.array(roi, dtype='int32')
#.......................................................................................

    
def gdist_label(source_labels, dlabel, surface, hemisphere, mean=False):
    '''
    Compute geodesic distance from one or more labels
        source_labels: label of the parcel of which you need the nodes (must match parcel names in dlabel file)
        dlabel: reference cifti2.dlabel file
        surface: surface file
        hemisphere: 'L'/'R'
        mean: calculate the mean geodesic distance, if False calculate shortest
    '''
    brain_structure = ('CIFTI_STRUCTURE_CORTEX_LEFT' if hemisphere=='L' else 'CIFTI_STRUCTURE_CORTEX_RIGHT')
    cortex = np.array(struct_to_vtx(brain_structure, dlabel), dtype='int32')[0]
    vertices, triangles = vtx_to_surf(cortex, surface)
    vertices = np.array(surface.darrays[0].data,dtype='float64')
    triangles = np.array(surface.darrays[1].data,dtype='int32')
    
    dist = []
    for source in source_labels:
        source = np.int32(label_to_vtx(source, dlabel, brain_structure)[1])
        
        if mean:
            dist_lab = np.zeros(len(cortex))
            for vtx in source:
                dist_vtx = gdist.compute_gdist(vertices, triangles, source_indices=vtx, target_indices=cortex)
                dist_lab += dist_vtx
            dist_lab = dist_lab/len(source)
            dist.append(dist_lab)
        else:
            dist_lab = gdist.compute_gdist(vertices, triangles, source_indices=source, target_indices=cortex)
            dist.append(dist_lab)
            
    return dist
#.......................................................................................


### Smooth time series using wb_command
def smooth_dtseries(in_dtseries, out_dtseries, kernel, L_surface, R_surface):
    '''
    Python wrapper of the wb_command -cifti-smoothing command
        in_dtseries: path to the dtseries to smooth
        out_dtseries: path and name of the output smoothed dtseries
        kernel: smoothing kernel (fwhm)
        L_surface: path to the left surface file
        R_surface: path to the right surface file
    '''
    sp.run(f'wb_command -cifti-smoothing {in_dtseries} -fwhm {kernel} {kernel} \
    COLUMN {out_dtseries} -left-surface {L_surface} -right-surface {R_surface}',shell=True)
#.......................................................................................


def save_dscalar(scalars, template_cifti, output_dir, names=None):
    '''
    Save 2D np.array as dscalar file
        data: MxN np.array
        template_cifti: any cifti2 file with a BrainModelAxis with N vertices
        output_dir: full path and name of the output file
        names: list with M names corresponding to the metric stored in each row of scalars
    '''
    
    if scalars.ndim==1:
        scalars = scalars.reshape(1, scalars.size)
        
    data = np.zeros([scalars.shape[0],template_cifti.shape[1]])
    data[0:,0:scalars.shape[1]] = scalars
    
    if names == None:
        map_labels = np.arange(scalars.shape[0])+1
    else:
        map_labels = names
    ax0 = nib.cifti2.cifti2_axes.ScalarAxis(map_labels)
    ax1 = nib.cifti2.cifti2_axes.from_index_mapping(template_cifti.header.get_index_map(1))
    nifti_hdr = template_cifti.nifti_header
    del template_cifti
    
    new_img = nib.Cifti2Image(data, header=[ax0, ax1],nifti_header=nifti_hdr)
    new_img.update_headers()

    new_img.to_filename(output_dir)
#.......................................................................................
