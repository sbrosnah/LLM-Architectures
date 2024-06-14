import torch 
import torch.nn as nn
from torch.utils.data import Dataset 
from config import TrainingConfig

from torch.utils.data import Dataset, DataLoader, random_split

from tokenizers import Tokenizer

from utils import causal_mask, padding_mask, ExtraTokens
from shared import EncoderDecoderBatchKeys, DecoderOnlyBatchKeys

from typing import Callable, Any

    
class DatasetPreprocessor:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
    
    def calc_train_test_size(self, data):
        train_ds_size = int((1 - self.config.VAL_PERCENTAGE) * len(data))
        val_ds_size = len(data) - train_ds_size
        return train_ds_size, val_ds_size
            
    def build_dataloaders(self, ds_raw: Dataset, get_dataset: Callable[[Any], Any]) -> tuple[DataLoader, DataLoader]:

        #Keep 90% for training and 10% for validation
        train_ds_size, val_ds_size = self.calc_train_test_size(ds_raw)
        train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

        train_ds = get_dataset(train_ds_raw)
        val_ds = get_dataset(val_ds_raw)

        train_dataloader = DataLoader(train_ds, batch_size=self.config.BATCH_SIZE, shuffle=True)
        val_dataloader = DataLoader(val_ds, batch_size=self.config.BATCH_SIZE, shuffle=True)

        return train_dataloader, val_dataloader
    
    def chunk_text(self, text):
        train_size, val_size = self.calc_train_test_size(text)
        

class EncoderDecoderDataset(Dataset):

    def __init__(self, ds: Dataset, config: TrainingConfig, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer, 
                get_src_text: Callable[[Any], Any], get_tgt_text: Callable[[Any], Any]) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.config = config
        self.get_src_text = get_src_text
        self.get_tgt_text = get_tgt_text

        #These special tokens will be the same in both languages
        self.sos_token = torch.tensor([tokenizer_src.token_to_id(ExtraTokens.SOS_TOKEN)], dtype=torch.int64) #use int64 because anything smaller may not suffice for the vocab size
        self.eos_token = torch.tensor([tokenizer_src.token_to_id(ExtraTokens.EOS_TOKEN)], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id(ExtraTokens.PAD_TOKEN)], dtype=torch.int64)
    
    def __len__(self) -> int:
        return len(self.ds)
    
    def __getitem__(self, index: any) -> any:
        src_target_pair = self.ds[index]
        src_text = self.get_src_text(src_target_pair)
        tgt_text = self.get_tgt_text(src_target_pair)

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.config.SEQ_LEN - len(enc_input_tokens) - 2 #subtract 2 because we will have eos and sos
        dec_num_padding_tokens = self.config.SEQ_LEN - len(dec_input_tokens) - 1 #subtract 1 because we will have eos

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        #Add SOS and EOS to the source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0
        )

        #Add SOS to the decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        # Add EOS to the label (what we expect as output from the decoder)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        assert encoder_input.size(0) == self.config.SEQ_LEN
        assert decoder_input.size(0) == self.config.SEQ_LEN
        assert label.size(0) == self.config.SEQ_LEN
        
        encoder_mask = padding_mask(encoder_input, self.pad_token)
        decoder_mask = padding_mask(decoder_input, self.pad_token) & causal_mask(decoder_input.size(0))

        return {
            EncoderDecoderBatchKeys.ENCODER_INPUT: encoder_input, 
            EncoderDecoderBatchKeys.DECODER_INPUT: decoder_input, 
            EncoderDecoderBatchKeys.LABEL: label,
            EncoderDecoderBatchKeys.ENCODER_MASK: encoder_mask,
            EncoderDecoderBatchKeys.DECODER_MASK: decoder_mask,
            EncoderDecoderBatchKeys.SRC_TEXT: src_text,
            EncoderDecoderBatchKeys.TGT_TEXT: tgt_text
        }

class DecoderOnlyDataset(Dataset):

    def __init__(self, text: Dataset, config: TrainingConfig, tokenizer: Tokenizer) -> None:
        super().__init__()

        self.text = text
        self.tokenizer = tokenizer
        self.config = config

        #These special tokens will be the same in both languages
        self.sos_token = torch.tensor([tokenizer.token_to_id(ExtraTokens.SOS_TOKEN)], dtype=torch.int64) #use int64 because anything smaller may not suffice for the vocab size
        self.eos_token = torch.tensor([tokenizer.token_to_id(ExtraTokens.EOS_TOKEN)], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer.token_to_id(ExtraTokens.PAD_TOKEN)], dtype=torch.int64)
    
    def __len__(self) -> int:
        return len(self.ds)
    
    def __getitem__(self, index: any) -> any:
        input_text = self.ds[index]

        dec_input_tokens = self.tokenizer.encode(input_text).ids

        dec_num_padding_tokens = self.config.SEQ_LEN - len(dec_input_tokens) - 1 #subtract 1 because we will have eos

        #Add SOS to the decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        # Add EOS to the label (what we expect as output from the decoder)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        assert decoder_input.size(0) == self.config.SEQ_LEN
        assert label.size(0) == self.config.SEQ_LEN
        
        decoder_mask = padding_mask(decoder_input, self.pad_token) & causal_mask(decoder_input.size(0))

        return {
            DecoderOnlyBatchKeys.DECODER_INPUT: decoder_input,
            DecoderOnlyBatchKeys.DECODER_MASK: decoder_mask,
            DecoderOnlyBatchKeys.LABEL: label,
            DecoderOnlyBatchKeys.INPUT_TEXT: input_text
        }



    
        

