TRAINING_EPOCH = 500
NUM_CLASSES = 1

RESIZE_DEPTH = 64
RESIZE_HEIGHT = 128
RESIZE_WIDTH = 128


### 치아에 대한 ISO 기준 Numbering ###
UPPER_TOOTH_NUM = [
    '21', '22', '23', '24', '25', '26', '27', '28',
    '11', '12', '13', '14', '15', '16', '17', '18', 
]

LOWER_TOOTH_NUM = [
    '31', '32', '33', '34', '35', '36', '37', '38',
    '41', '42', '43', '44', '45', '46', '47', '48',
]


### OSSTEM Dataset에서 개별 치아에 대한 Detection을 위한 Train / Test 분리 ###
DETECTION_TRAIN = [
    '1',  '2',  '3',  '4',  '5',
    '6',  '7', 
          '12', '13', '14', '15',
    '16', '17', '18', '19',
    '21', '22', '23', '24',
    '26', '27', '28', '29', '30',
                      '34', '35',
                '38',       
          '42',             '45',
                            '50',
]

DETECTION_TEST = [
    '37','41','46','47','48','8','9','40'
]


### OSSTEM Dataset에서 개별 치아에 대한 Metal Classification 을 위한 Train / Test 분리 ###
METAL_TRAIN = [
    '2', '3', '4', '5',
    '6', '7', 
                      '14', '15',
          '17', '18',
          '22', '23',
                      '29', '30',
                      '34',
          '37', '38',       

    '46', '47',             '50',
]

METAL_TEST = [
    '1', '41', '8', '12', '13', '28', '35', '42',
]


### OSSTEM Dataset 한장씩 봤을 때, 영상과 Segmentation Annotation이 달라 Box, Center GT랑 일치하지 않는 Noisy한 치아 Case ###
OUTLIER = {'1' : ['34'],
           '2' : ['12', '13'],
           '3' : ['17', '34', '36', '42', '43', '44', '45', '47'],
           '4' : ['16', '23', '25', '35', '37', '42', '47'],
           '5' : ['35'],
           '10' : ['11'],
           '11' : ['11'],
           '14' : ['31'],
           '16' : ['42'],
           '17' : ['43'],
           '18' : ['13'],
           '22' : ['33'],
           '24' : ['13'],
           '27' : ['45'],
           '40' : ['16'],
           '42' : ['46'],
           '45' : ['25']}

### Focal Loss 의 Positive 정의를 위한 Epsilon ###
FOCAL_EPSILON = 1e-3

### DR Loss 주기 시작하는 Epoch ###
DR_EPOCH = 40

### Metal BCE Loss 주기 시작하는 Epoch ###
METAL_EPOCH = 40

### Metal Classification Label Smoothing을 위한 Epsilon ###
METAL_EPSILON = 0.05