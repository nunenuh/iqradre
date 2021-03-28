import fire
import torch
import logging
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from iqradre.extract.data import utils 
from iqradre.extract.data import loader
from iqradre.extract.config import label as label_cfg
from iqradre.extract.trainer.task import TaskLayoutLM

from transformers import BertTokenizer, AutoModel
from transformers import (
    WEIGHTS_NAME,
    BertTokenizer,
    LayoutLMConfig,
    LayoutLMForTokenClassification,
)


# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train(dataset_path, 
          loader_type='combined_loader',
          batch_size=4, num_workers=16,
          dataset_rand_seq=True, 
          dataset_rand_seq_prob=0.5, 
          dataset_exlude_image_mask=True,
          
          state_dict_path=None,
          weight_path = 'weights/extract/',
          
          
          max_epoch=5,
          lr=0.001, 
          valcheck_interval=2000,
          num_gpus=1, log_freq=100,
          resume_checkpoint_path = None,
          checkpoint_saved_path = "checkpoints/v2/",
          logs_path = "logs/v2/",
          prefix_name = 'layoutlm-v2',
          manual_seed=1261):

    logging.basicConfig(level=logging.INFO)
    
    #load tokensizer
    logging.info("Load BertTokenizer with indobenchmark/indobert-base-p2")
    tokenizer = BertTokenizer.from_pretrained(
        "indobenchmark/indobert-base-p2",
        do_lower_case=True,
        cache_dir=None,
    )

    path = dataset_path
    if loader_type=='combined_loader':
        logging.info(f"Load Combined Loader with path {dataset_path}")
        train_loader, valid_loader = loader.get_loader(
            path, tokenizer=tokenizer, 
            batch_size=batch_size, 
            num_workers=num_workers,
            rand_seq=dataset_rand_seq,
            rand_seq_prob=dataset_rand_seq_prob,
            excluce_image_mask=dataset_exlude_image_mask
        )
    else:
        logging.info(f"Load Base Loader with path {dataset_path}")
        train_loader, valid_loader = loader.get_base_loader(
            path, tokenizer=tokenizer, 
            batch_size=batch_size, 
            num_workers=num_workers,
            rand_seq=dataset_rand_seq,
            rand_seq_prob=dataset_rand_seq_prob,
            excluce_image_mask=dataset_exlude_image_mask
        )

    logging.info(f"Load LayoutLMConfig for LayoutLMForTokenClassification")
    config = LayoutLMConfig.from_pretrained(
        "microsoft/layoutlm-base-uncased",
        num_labels=label_cfg.num_labels,
        cache_dir=None
    )

    logging.info(f"Load LayoutLMForTokenClassification from_pretrained microsoft/layoutlm-base-uncased")
    model = LayoutLMForTokenClassification.from_pretrained(
        'microsoft/layoutlm-base-uncased',
        config=config,
    #     return_dict=True
    )
    model.resize_token_embeddings(len(tokenizer))
    
    
    if state_dict_path:
        logging.info(f"Load state_dict from path {state_dict_path}")
        state_dict = torch.load(state_dict_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        # model = model.to(device)

    #prepare the task
    task = TaskLayoutLM(model, tokenizer)

    # DEFAULTS used by the Trainer
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=checkpoint_saved_path,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=prefix_name
    )

    tb_logger = pl_loggers.TensorBoardLogger(logs_path)
    pl.trainer.seed_everything(manual_seed)

    trainer = pl.Trainer(
        weights_summary="top",
        max_epochs=max_epoch,
        val_check_interval=valcheck_interval,
        gpus=num_gpus,
        log_every_n_steps=log_freq,
        deterministic=True,
        benchmark=True,
        logger=tb_logger, 
        checkpoint_callback=checkpoint_callback, 
        resume_from_checkpoint=resume_checkpoint_path,
    )
    trainer.fit(task, train_loader, valid_loader)
    # metrics = trainer.test(task, valid_loader)
    print('Metircs', trainer.logged_metrics)

    #prepare to save result
    # print(task._results.keys())
    vacc , vloss = task._results['val_acc'], task._results['val_loss']
    dirname = Path(dataset_path).name
    filename = f'layoutlm_v2_ktp_{dirname}_vacc{vacc:.4}_vloss{vloss:.4}_cli.pth'
    saved_filename = str(Path(weight_path).joinpath(filename))
    
    logging.info(f"Prepare to save training results to path {saved_filename}")
    torch.save(model.state_dict(), saved_filename)


if __name__ == '__main__':
  fire.Fire(train)
