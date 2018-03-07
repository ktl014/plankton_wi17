IMG_NAME = 'images'
CLASS = 'class'
COORDINATES = 'coordinates'

ALEXNET = 'alexnet'
RESNET50 = 'resnet50'
VGG16 = 'vgg16'
VGG19 = 'vgg19'
MODELS = [ALEXNET, RESNET50, VGG16]

HEAD_X = 'head_x_rel'
HEAD_Y = 'head_y_rel'
TAIL_X = 'tail_x_rel'
TAIL_Y = 'tail_y_rel'
COORDINATES_LIST = [HEAD_X, HEAD_Y, TAIL_X, TAIL_Y]

ORDER = 'order'
FAMILY = 'family'
GENUS = 'genus'
SPECIMEN_ID = 'specimen_id'
CLASS_LIST = [ORDER, FAMILY, GENUS, SPECIMEN_ID]

TRAIN = 'train'
VALID = 'valid'
TEST = 'test'
PHASES = [TRAIN, VALID, TEST]


class GpuMode:
    SINGLE = 'single'
    MULTI = 'multi'
    CPU = 'cpu'
