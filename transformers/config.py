class TrainingConfig:
    BATCH_SIZE = 8
    NUM_EPOCHS = 30
    LR = 10**-4
    SEQ_LEN = 800
    D_MODEL = 512
    MODEL_FOLDER = "transformers/weights"
    MODEL_BASENAME = "tmodel_"
    PRELOAD = None
    TOKENIZER_FILE = "transformers/tokenizer/tokenizer_{0}.json"
    EXPERIMENT_NAME = "transformers/runs/tmodel"
    EVAL_EVERY = 200
    VAL_PERCENTAGE = .1
    EVAL_ITERS = 10
    LABEL_SMOOTHING = 0.1
    
class ModelConfig:
    D_MODEL = 512
    D_FF = 2048
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    NUM_HEADS = 8
    D_Q_K = 64
    D_V = 64
    DROPOUT = 0.1