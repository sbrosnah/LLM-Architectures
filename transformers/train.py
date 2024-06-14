import torch
import torch.nn as nn

from model import Transformer
from config import TrainingConfig, ModelConfig

from utils import causal_mask, ExtraTokens, get_max_seq_length, get_or_build_tokenizer, determine_device
from shared import TrainingStateKeys, EncoderDecoderBatchKeys, BatchKeys

from tokenizers import Tokenizer

from datasets import load_dataset
from dataset import DatasetPreprocessor, EncoderDecoderDataset
from torch.utils.data import DataLoader
from typing import Callable, Any

from torch.utils.tensorboard import SummaryWriter

import warnings
from pathlib import Path

from tqdm import tqdm

from typing import Generator
from tokenizers import Tokenizer

class TextGenerator:
    def __init__(self, config: TrainingConfig, device: torch.device, tokenizer: Tokenizer) -> None:
        self.config = config
        self.device = device
        self.tokenizer = tokenizer
    
    def greedy_decode(self, model: Transformer, source: torch.Tensor, source_mask: torch.Tensor) -> str:
        
        sos_idx = self.tokenizer.token_to_id(ExtraTokens.SOS_TOKEN)
        eos_idx = self.tokenizer.token_to_id(ExtraTokens.EOS_TOKEN)
        
        # Precompute the eencoder output and reuse it for every token we get from the decoder 
        encoder_output = model.encode(source, source_mask)
        #Initialize the decoder input with the sos token 
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(self.device)
        while True:
            
            if decoder_input.size(1) == self.config.SEQ_LEN:
                break
            
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(self.device)
        
            out = model.decode(decoder_input, decoder_mask, encoder_output, source_mask)
            
            #we only want the projection of the last token
            prob = model.project(out[:, -1])
            
            #greedy search
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(self.device)], dim=1)
            
            if next_word == eos_idx:
                break
            
        model_out = decoder_input.squeeze(0)
        
        return self.tokenizer.decode(model_out.detach().cpu().numpy()) 

class Trainer():
    def __init__(self, config: TrainingConfig, device: torch.device) -> None:
        self.config = config
        self.device = device      
    
    def get_weights_file_path(self, epoch) -> str:
        model_folder = self.config.MODEL_FOLDER
        model_basename = self.config.MODEL_BASENAME
        model_filename = f"{model_basename}{epoch}.pt"
        return str(Path('.') / model_folder / model_filename)
    
    #This is only used in encoder-decoder tasks
    def generate_examples(self, model: Transformer, batch: Any, print_msg: Callable[[str], None], generator: TextGenerator, 
                          num_examples: int, writer: SummaryWriter, global_step: int) -> None:
        
        
        source_texts = []
        expected = []
        predicted = []
        
        console_width = 80
        
        encoder_input = batch[EncoderDecoderBatchKeys.ENCODER_INPUT].to(self.device)
        source_mask = batch[EncoderDecoderBatchKeys.ENCODER_MASK].to(self.device)
        
        assert num_examples < encoder_input.size(0), "num examples must be less than batch size"
        
        for i in range(num_examples):
            example_index = i
            

            encoder_in = encoder_input[example_index].unsqueeze(0)
            src_mask = source_mask[example_index].unsqueeze(0)
            
            assert encoder_in.size(0) == 1, "Batch size must be 1 for generation"
            
            #TODO: make the decode strategy configurable
            model_out_text = generator.greedy_decode(model, encoder_in, src_mask)
            
            
            source_text = batch[EncoderDecoderBatchKeys.SRC_TEXT][example_index][0]
            target_text = batch[EncoderDecoderBatchKeys.TGT_TEXT][example_index][0]
            
            
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            print_msg('_'*console_width)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')
        
        if writer: 
            # TorchMetrics CharErrorRate, BLEU, WordErrorRate
            pass
    

    def run_validation(self, model: Transformer, tokenizer: Tokenizer, get_model_input: Callable, loss_fn: Any, validation_ds: DataLoader, print_msg: Callable[[str], None], 
                       global_step: int, writer: SummaryWriter, generator: TextGenerator, num_examples:int=2) -> None:
        
        model.eval()
        
        with torch.no_grad():
            
            validation_losses = torch.zeros(self.config.EVAL_ITERS)
            
            for i, batch in enumerate(validation_ds):
                
                if i >= self.config.EVAL_ITERS:
                    break
                
                if i == 0:
                    self.generate_examples(model, batch, print_msg, generator, num_examples, writer, global_step)
                
                #Do the actual loss calculations 
                model_input = get_model_input(batch)
                
                #Run the tensors through the transformer
                out = model(*model_input) # (batch, seq_len, d_model)
                
                label = batch[BatchKeys.LABEL].to(self.device) #(batch, seq_len)
                
                # (batch, seq_len, tgt_vocab_size) --> (batch * seq_len, tgt_vocab_size)
                loss = loss_fn(out.view(-1, tokenizer.get_vocab_size()), label.view(-1))   
                
                validation_losses[i] = loss.item()
        
        return validation_losses.mean()
    
    def train_model(self, model: Transformer, train_dataloader: DataLoader, val_dataloader: DataLoader, 
                    tokenizer: Tokenizer, get_model_input: Callable) -> None:
        
        print(f"using device {self.device}")

        Path(self.config.MODEL_FOLDER).mkdir(parents=True, exist_ok=True)
        
        generator = TextGenerator(config, device, tokenizer)

        #Tensorboard for visualizing the loss 
        writer = SummaryWriter(self.config.EXPERIMENT_NAME)

        #TODO: make the optimizer configurable
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.LR, eps=1e-9)

        initial_epoch = 0
        global_step = 0
        if self.config.PRELOAD:
            model_filename = self.get_weights_file_path(self.config.PRELOAD)
            print(f"Preloading model {model_filename}")
            state = torch.load(model_filename)
            model.load_state_dict(state[TrainingStateKeys.MODEL_STATE_DICT])
            initial_epoch = state[TrainingStateKeys.EPOCH] + 1
            optimizer.load_state_dict(state[TrainingStateKeys.OPTIMIZER_STATE_DICT])
            global_step = state[TrainingStateKeys.GLOBAL_STEP]
        
        #label smoothing spreads the distribution which improves the accuracy. .1 means we take .1 from the highest score
        #and distribute it in the others
        loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id(ExtraTokens.PAD_TOKEN), label_smoothing=self.config.LABEL_SMOOTHING).to(self.device)
        
        train_loss = None 
        val_loss = None
        
        for epoch in range(initial_epoch, self.config.NUM_EPOCHS):
            
            batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
            for batch in batch_iterator:
                
                model.train()
                
                model_input = get_model_input(batch)
                
                #Run the tensors through the transformer
                out = model(*model_input) # (batch, seq_len, d_model)
                
                label = batch[BatchKeys.LABEL].to(self.device) #(batch, seq_len)
                
                # (batch, seq_len, tgt_vocab_size) --> (batch * seq_len, tgt_vocab_size)
                loss = loss_fn(out.view(-1, tokenizer.get_vocab_size()), label.view(-1))
                train_loss = loss.item()
                
                if global_step % self.config.EVAL_EVERY == 0:
                    
                    mean_val_loss = self.run_validation(
                        model,
                        tokenizer,
                        get_model_input,
                        loss_fn, 
                        val_dataloader,
                        lambda msg: batch_iterator.write(msg), 
                        global_step, 
                        writer, 
                        generator)
                    
                    val_loss = mean_val_loss.item()
                    
                    writer.add_scalar('val loss', val_loss, global_step)
                    writer.flush()
                
                
                #log the loss
                writer.add_scalar('train loss', train_loss, global_step)
                writer.flush()
                    
                batch_iterator.set_postfix({f"train loss": f"{train_loss:6.3f}", 
                                           f"val loss": f"{val_loss:6.3f}"})
                    
                
                loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()
                
                global_step += 1
            
            #Save the model after each epoch 
            model_filename = self.get_weights_file_path(f'{epoch:02d}')
            
            torch.save({
                TrainingStateKeys.EPOCH: epoch,
                TrainingStateKeys.MODEL_STATE_DICT: model.state_dict(),
                TrainingStateKeys.OPTIMIZER_STATE_DICT: optimizer.state_dict(),
                TrainingStateKeys.GLOBAL_STEP: global_step
            }, model_filename)



if __name__=='__main__':
    
    warnings.filterwarnings("ignore")
    
    device = determine_device()
    config = TrainingConfig
    trainer = Trainer(config, device)
    
    # LANG_SRC = "en"
    # LANG_TGT = "es"
    
    # ds_raw = load_dataset('opus_books', f'{LANG_SRC}-{LANG_TGT}', split='train')

    # pre_processor = DatasetPreprocessor(config)
    # #TODO: This is where I will split data for a decoder-only training run
    
    # def get_all_sentences(ds, lang) -> Generator[str, None, None]:
    #     for item in ds:
    #         yield item['translation'][lang]
    
    # tokenizer_src = get_or_build_tokenizer(
    #     get_all_sentences(ds_raw, LANG_SRC), 
    #     Path(config.TOKENIZER_FILE.format(LANG_SRC)))
    
    # tokenizer_tgt = get_or_build_tokenizer(
    #     get_all_sentences(ds_raw, LANG_TGT), 
    #     Path(config.TOKENIZER_FILE.format(LANG_TGT)))
    
    # max_len_src = get_max_seq_length(ds_raw, tokenizer_src, lambda x: x['translation'][LANG_SRC])
    # max_len_tgt = get_max_seq_length(ds_raw, tokenizer_tgt, lambda x: x['translation'][LANG_TGT])
    # print(f"Max length of source sentence: {max_len_src}")
    # print(f"Max length of target sentence: {max_len_tgt}")
    
    # def get_dataset(ds):
    #     return EncoderDecoderDataset(ds, config, tokenizer_src, tokenizer_tgt,
    #                           lambda x: x['translation'][LANG_SRC],
    #                           lambda x: x['translation'][LANG_TGT])
    
    # train_dataloader, val_dataloader = pre_processor.build_dataloaders(ds_raw, get_dataset)
    
    # model_config = ModelConfig
    
    # model = Transformer(tokenizer_src.get_vocab_size(), 
    #                 tokenizer_tgt.get_vocab_size(), 
    #                 config.SEQ_LEN,  
    #                 model_config).to(device)
    
    # def get_model_input(batch: any) -> tuple[any, any, any, any]:
    #     decoder_input = batch[EncoderDecoderBatchKeys.DECODER_INPUT].to(device) #(batch, seq_len)
    #     decoder_mask = batch[EncoderDecoderBatchKeys.DECODER_MASK].to(device) #(batch, 1, seq_len, seq_len)
    #     encoder_input = batch[EncoderDecoderBatchKeys.ENCODER_INPUT].to(device) #(batch, seq_len)
    #     encoder_mask = batch[EncoderDecoderBatchKeys.ENCODER_MASK].to(device) # (batch, 1, 1, seq_len)
    #     return (decoder_input, decoder_mask, encoder_input, encoder_mask)
    
    # trainer.train_model(model, train_dataloader, val_dataloader, tokenizer_tgt, get_model_input)
    
    with open('transformers/notebooks/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    pre_processor = DatasetPreprocessor(config)
    #TODO: This is where I will split data for a decoder-only training run
    text_chunks = pre_processor.chunk_corpus(text)
    
    def get_all_sentences(ds, lang) -> Generator[str, None, None]:
        for item in ds:
            yield item['translation'][lang]
    
    tokenizer_src = get_or_build_tokenizer(
        get_all_sentences(ds_raw, LANG_SRC), 
        Path(config.TOKENIZER_FILE.format(LANG_SRC)))
    
    tokenizer_tgt = get_or_build_tokenizer(
        get_all_sentences(ds_raw, LANG_TGT), 
        Path(config.TOKENIZER_FILE.format(LANG_TGT)))
    
    max_len_src = get_max_seq_length(ds_raw, tokenizer_src, lambda x: x['translation'][LANG_SRC])
    max_len_tgt = get_max_seq_length(ds_raw, tokenizer_tgt, lambda x: x['translation'][LANG_TGT])
    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")
    
    def get_dataset(ds):
        return EncoderDecoderDataset(ds, config, tokenizer_src, tokenizer_tgt,
                              lambda x: x['translation'][LANG_SRC],
                              lambda x: x['translation'][LANG_TGT])
    
    train_dataloader, val_dataloader = pre_processor.build_dataloaders(ds_raw, get_dataset)
    
    model_config = ModelConfig
    
    model = Transformer(tokenizer_src.get_vocab_size(), 
                    tokenizer_tgt.get_vocab_size(), 
                    config.SEQ_LEN,  
                    model_config).to(device)
    
    def get_model_input(batch: any) -> tuple[any, any, any, any]:
        decoder_input = batch[EncoderDecoderBatchKeys.DECODER_INPUT].to(device) #(batch, seq_len)
        decoder_mask = batch[EncoderDecoderBatchKeys.DECODER_MASK].to(device) #(batch, 1, seq_len, seq_len)
        encoder_input = batch[EncoderDecoderBatchKeys.ENCODER_INPUT].to(device) #(batch, seq_len)
        encoder_mask = batch[EncoderDecoderBatchKeys.ENCODER_MASK].to(device) # (batch, 1, 1, seq_len)
        return (decoder_input, decoder_mask, encoder_input, encoder_mask)
    
    trainer.train_model(model, train_dataloader, val_dataloader, tokenizer_tgt, get_model_input)