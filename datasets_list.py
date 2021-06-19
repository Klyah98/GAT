import datasets.cora as cora

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
}
