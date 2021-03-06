from detectron2.config import CfgNode as CN

INF = 1e8

def add_querydet_config(cfg):
    cfg.MODEL.FPN.TOP_LEVELS = 2
    
    #----------------------------------------------------------------------------------------------
    #                                      CUSTOM
    #----------------------------------------------------------------------------------------------
    cfg.MODEL.CUSTOM = CN()

    cfg.MODEL.CUSTOM.FOCAL_LOSS_GAMMAS = []
    cfg.MODEL.CUSTOM.FOCAL_LOSS_ALPHAS = []

    cfg.MODEL.CUSTOM.CLS_WEIGHTS = []
    cfg.MODEL.CUSTOM.REG_WEIGHTS = []
    
    cfg.MODEL.CUSTOM.USE_LOOP_MATCHER = False
    
    # soft nms
    cfg.MODEL.CUSTOM.USE_SOFT_NMS       = False
    cfg.MODEL.CUSTOM.GIOU_LOSS          = False
    cfg.MODEL.CUSTOM.SOFT_NMS_METHOD    = 'linear' # gaussian
    cfg.MODEL.CUSTOM.SOFT_NMS_SIGMA     = 0.5
    cfg.MODEL.CUSTOM.SOFT_NMS_THRESHOLD = 0.5
    cfg.MODEL.CUSTOM.SOFT_NMS_PRUND     = 0.001 

    cfg.MODEL.CUSTOM.HEAD_BN = False
    
    #----------------------------------------------------------------------------------------------
    #                                          QUERY
    #----------------------------------------------------------------------------------------------
    cfg.MODEL.QUERY = CN()

    cfg.MODEL.QUERY.FEATURES_WHOLE_TRAIN = [2, 3, 4, 5]
    cfg.MODEL.QUERY.FEATURES_VALUE_TRAIN = [0, 1]
    cfg.MODEL.QUERY.Q_FEATURE_TRAIN = [2]

    cfg.MODEL.QUERY.FEATURES_WHOLE_TEST = [2, 3, 4, 5]
    cfg.MODEL.QUERY.FEATURES_VALUE_TEST = [0, 1]
    cfg.MODEL.QUERY.Q_FEATURE_TEST = [2]

    cfg.MODEL.QUERY.QUERY_LOSS_WEIGHT = []
    cfg.MODEL.QUERY.QUERY_LOSS_GAMMA  = []

    cfg.MODEL.QUERY.ENCODE_CENTER_DIS_COEFF = [1.]
    cfg.MODEL.QUERY.ENCODE_SMALL_OBJ_SCALE = []

    cfg.MODEL.QUERY.THRESHOLD = 0.12
    cfg.MODEL.QUERY.CONTEXT = 2

    cfg.MODEL.QUERY.QUERY_INFER = False
    
    #----------------------------------------------------------------------------------------------
    #                              APEX Mixed Precision Trianing
    #----------------------------------------------------------------------------------------------
    cfg.MODEL.APEX = CN()
    cfg.MODEL.APEX.ENABLE = False
    cfg.MODEL.APEX.LEVEL = 'O1'

    #----------------------------------------------------------------------------------------------
    #                                      Meta Info
    #----------------------------------------------------------------------------------------------
    cfg.META_INFO = CN()

    cfg.META_INFO.VIS_ROOT = ''
    cfg.META_INFO.EVAL_GPU_TIME = False
    cfg.META_INFO.EVAL_AP = True
    cfg.META_INFO.CLEAR_CUDA_CACHE = False

    #----------------------------------------------------------------------------------------------
    #                                      VisDrone2018
    #----------------------------------------------------------------------------------------------
    cfg.VISDRONE = CN()
    
    cfg.VISDRONE.TRAIN_JSON     = 'path/to/label.json'
    cfg.VISDRONE.TRING_IMG_ROOT = 'path/to/images'
    
    cfg.VISDRONE.TEST_JSON      = 'path/to/label.json'
    cfg.VISDRONE.TEST_IMG_ROOT  = 'path/to/images'

    cfg.VISDRONE.SHORT_LENGTH   = [1200]
    cfg.VISDRONE.MAX_LENGTH     = 1999

    cfg.VISDRONE.TEST_LENGTH   = 3999

