import torch
from torch.utils.data import Dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace 

from typing import Generator, Callable

from pathlib import Path

from shared import ExtraTokens

def determine_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def padding_mask(x: torch.Tensor, pad_token: torch.Tensor) -> torch.Tensor:
    return (x != pad_token).unsqueeze(0).unsqueeze(0).int()

def causal_mask(size: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0

def get_or_build_tokenizer(text: Generator[str, None, None], tokenizer_path: str) -> Tokenizer:
    
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token=ExtraTokens.UNK_TOKEN))
        tokenizer.pre_tokenizer = Whitespace() # means that we split by whitespace
        #This is what is used to actually build the tokenizer
        trainer = WordLevelTrainer(special_tokens=[ExtraTokens.UNK_TOKEN, ExtraTokens.PAD_TOKEN, ExtraTokens.SOS_TOKEN, ExtraTokens.EOS_TOKEN], min_frequency=2)
        tokenizer.train_from_iterator(text, trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
    return tokenizer

def get_max_seq_length(ds_raw: Dataset, tokenizer: Tokenizer, get_text: Callable):
    
    max_len = 0

    for item in ds_raw:
        ids = tokenizer.encode(get_text(item)).ids
        max_len = max(max_len, len(ids))
            
    return max_len