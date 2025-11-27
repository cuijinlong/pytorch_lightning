from typing import Any, Dict, List, Optional, Tuple
import sys
import hydra
#import lightning as L
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer,Callback
from pytorch_lightning.loggers import Logger
#from lightning import Callback, LightningDataModule, LightningModule, Trainer
#from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import torch.nn.functional as F
import torch.nn as nn
import os
import sys
from lightning.pytorch.tuner import Tuner
import logging

from trainer.lhgnn.utils.tool import (
    extras,
    get_metric_value,
    task_wrapper,
)



@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    log.info(f"Instantiating model <{cfg.model._target_}>")

    model: LightningModule = hydra.utils.instantiate(cfg.model)
    
    pretrained = cfg.get('pretrained')

    if pretrained in ['audioset','img']:
        log.info("Loading {} pretrained weights".format(pretrained))
    else:
        log.info(f"Training from scratch")


    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "callbacks": callbacks,
        "trainer": trainer,
    }

    trainer.fit(model=model, datamodule=datamodule)

    train_metrics = trainer.callback_metrics

    # Evaluate model on test set with best weights

    if cfg.get("eval"):
        path = '/data/scratch/acw572/runs/2024-04-13_14-38-01/checkpoints/last.ckpt'
        state_dict = torch.load(path)
        ckpt_path = trainer.checkpoint_callback.best_model_path
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(state_dict,strict=True)
        
        test_results = trainer.test(model=model, datamodule=datamodule)
        #logging.info(f"Test results: {test_results['mAP']}")
        #log.info(f"Test mAP: {test_results['mAP']}")
    # Weighted Average Model 
        
    if cfg.get("wa"):
        model_ckpt = []
        ckpt_dir = cfg.callbacks.get('model_checkpoint').get('dirpath')
        
        model: LightningModule = hydra.utils.instantiate(cfg.model)
        own_state = model.state_dict()
        for ckpt in os.listdir(ckpt_dir):

            if 'ckpt' in ckpt:
                #print(torch.load(os.path.join(ckpt_dir,ckpt))['state_dict'].keys())
                model_ckpt.append(torch.load(os.path.join(ckpt_dir,ckpt))['state_dict'])
        
        
        for name, params in own_state.items():
            own_state[name] = torch.zeros_like(params)
            model_ckpt_key = torch.cat([d[name].float().unsqueeze(0) for d in model_ckpt],dim=0)
            own_state[name].copy_(torch.mean(model_ckpt_key,dim=0))
        
        model.load_state_dict(own_state)
        
        test_results_wa = trainer.test(model=model, datamodule=datamodule)
        logging.info(f"Test mAP from weighted average model: {test_results_wa['mAP']}")
        torch.save(model.net.state_dict(),os.path.join(ckpt_dir,'wa.pth.tar'))
    
    metrics = trainer.callback_metrics

    # merge train and test metrics
    #metric_dict = {**train_metrics, **test_metrics}

    #return metric_dict, object_dict
    return metrics,object_dict



@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.
    
    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    print("In main")
    extras(cfg)
    metrics,_ = train(cfg)
    # train the model
    

    # safely retrieve metric value for hydra-based hyperparameter optimization
    #metric_value = get_metric_value(
     #   metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    #)

    # return optimized metric
   # return metric_value


if __name__ == "__main__":
    
    torch.set_float32_matmul_precision("high")
    main()
