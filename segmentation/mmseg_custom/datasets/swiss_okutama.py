# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class SwissOkutamaDataset(CustomDataset):
    """Swiss Drone and Okutama Drone Datasets

    In segmentation map annotation for Swiss Okutama dataset, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    CLASSES = ('Background', 'Outdoor structures', 'Buildings', 'Paved ground', 'Non-paved ground', 
               'Train tracks', 'Plants', 'Wheeled vehicles', 'Water', 'People')

    PALETTE = [[0, 0, 0], [237, 237, 237], [181, 0, 0], [135, 135, 135],
               [189, 107, 0], [128, 0, 128], [31, 123, 22], [6, 0, 130],
               [0, 168, 255], [240, 255, 0]]
    # PALETTE = np.array([
    #     [0, 0, 0],      # Background
    #     [237, 237, 237],# Outdoor structures
    #     [181, 0, 0],    # Buildings
    #     [135, 135, 135],# Paved ground
    #     [189, 107, 0],  # Non-paved ground
    #     [128, 0, 128],  # Train tracks
    #     [31, 123, 22],  # Plants
    #     [6, 0, 130],    # Wheeled vehicles
    #     [0, 168, 255],  # Water
    #     [240, 255, 0]   # People
    #     ])

    def __init__(self, **kwargs):
        super(SwissOkutamaDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)
        
