class TrainingStateKeys:
    EPOCH = 'epoch'
    MODEL_STATE_DICT = 'model_state_dict'
    OPTIMIZER_STATE_DICT = 'optimizer_state_dict'
    GLOBAL_STEP = 'global_step'

class BatchKeys:
    LABEL = 'label'
    DECODER_INPUT = 'decoder_input'
    DECODER_MASK = 'decoder_mask'

class DecoderOnlyBatchKeys(BatchKeys):
    INPUT_TEXT = 'input_text'
    
class EncoderDecoderBatchKeys(BatchKeys):
    ENCODER_INPUT = 'encoder_input'
    ENCODER_MASK = 'encoder_mask'
    SRC_TEXT ='src_text'
    TGT_TEXT = 'tgt_text'

class ExtraTokens:
    PAD_TOKEN = '[PAD]'
    EOS_TOKEN = '[EOS]'
    SOS_TOKEN = '[SOS]'
    UNK_TOKEN = '[UNK]'