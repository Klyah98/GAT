from datasets import cora, pubmed, citeseer

datasets_list = {
    'cora': {
        'constants': (
            cora.TRAIN_RANGE,
            cora.VAL_RANGE,
            cora.TEST_RANGE,
            cora.NUM_INPUT_FEATURES,
            cora.NUM_CLASSES
        ),
        'load_function': cora.load_cora,
    },
    'pubmed': {
        'constants': (
            pubmed.TRAIN_RANGE,
            pubmed.VAL_RANGE,
            pubmed.TEST_RANGE,
            pubmed.NUM_INPUT_FEATURES,
            pubmed.NUM_CLASSES
        ),
        'load_function': pubmed.load_pubmed,
    },
    'citeseer': {
        'constants': (
            citeseer.TRAIN_RANGE,
            citeseer.VAL_RANGE,
            citeseer.TEST_RANGE,
            citeseer.NUM_INPUT_FEATURES,
            citeseer.NUM_CLASSES
        ),
        'load_function': citeseer.load_citeseer,
    },
}
