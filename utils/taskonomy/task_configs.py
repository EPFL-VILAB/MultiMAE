####################
# Tasks
####################

task_parameters = {
    'class_object':{
        'num_classes': 1000,
        'ext': 'npy',
        'domain_id': 'class_object',
    },
    'class_scene':{
        'num_classes': 365,
        'ext': 'npy',
        'domain_id': 'class_scene',
    },
    'depth_zbuffer':{
        'num_channels': 1,
        'mask_val': 1.0,
        'clamp_to': (0.0, 8000.0 / (2**16 - 1)), # Same as consistency
        'ext': 'png',
        'domain_id': 'depth_zbuffer',
    },
    'depth_euclidean':{
        'num_channels': 1,
        'clamp_to': (0.0, 8000.0 / (2**16 - 1)), # Same as consistency
#         'mask_val': 1.0,
        'ext': 'png',
        'domain_id': 'depth_euclidean',
    },
    'edge_texture': {
        'num_channels': 1,
        'clamp_to': (0.0, 0.25),
        #'threshold_min': 0.01,
        'ext': 'png',
        'domain_id': 'edge_texture',
    },
    'edge_occlusion': {
        'num_channels': 1,
        #'clamp_to': (0.0, 0.04),
        #'threshold_min': 0.0017,
        'ext': 'png',
        'domain_id': 'edge_occlusion',
    },
    'keypoints3d': {
        'num_channels': 1,
        'ext': 'png',
        'domain_id': 'keypoints3d',
    },
    'keypoints2d':{
        'num_channels': 1,
        #'clamp_to': (0.0, 0.025),
        #'threshold_min': 0.002,
        'ext': 'png',
        'domain_id': 'keypoints2d',
    },
    'principal_curvature':{
        'num_channels': 3,
        'mask_val': 0.0,
        'ext': 'png',
        'domain_id': 'principal_curvature',
    },
    'reshading':{
        'num_channels': 1,
        'ext': 'png',
        'domain_id': 'reshading',
    }, 
    'normal':{
        'num_channels': 3,
        'mask_val': 0.502,
        'ext': 'png',
        'domain_id': 'normal',
    },
    'mask_valid':{
        'num_channels': 1,
        'mask_val': 0.0,
        'ext': 'png',
        'domain_id': 'depth_zbuffer',
    },
    'rgb':{
        'num_channels': 3,
        'ext': 'png',
        'domain_id': 'rgb',
    },
    'segment_semantic': {
        'num_channels': 18,
        'ext': 'png',
        'domain_id': 'segmentsemantic',
    },
    'segment_unsup2d':{
        'num_channels': 64,
        'ext': 'png',
        'domain_id': 'segment_unsup2d',
    },
    'segment_unsup25d':{
        'num_channels': 64,
        'ext': 'png',
        'domain_id': 'segment_unsup25d',
    },
}

        
PIX_TO_PIX_TASKS = ['colorization', 'edge_texture', 'edge_occlusion',  'keypoints3d', 'keypoints2d', 'reshading', 'depth_zbuffer', 'depth_euclidean', 'curvature', 'autoencoding', 'denoising', 'normal', 'inpainting', 'segment_unsup2d', 'segment_unsup25d', 'segment_semantic', ]
FEED_FORWARD_TASKS = ['class_object', 'class_scene', 'room_layout', 'vanishing_point']
SINGLE_IMAGE_TASKS = PIX_TO_PIX_TASKS + FEED_FORWARD_TASKS
SIAMESE_TASKS = ['fix_pose', 'jigsaw', 'ego_motion', 'point_match', 'non_fixated_pose']
