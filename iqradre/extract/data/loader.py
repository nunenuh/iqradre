from .dataset import IDCardDataset, IDCardBaseDataset
from torch.utils.data.dataloader import DataLoader, RandomSampler, SequentialSampler

# from laylm.data.dataset import IDCardDataset

def get_base_dataloader(path, tokenizer, mode="train", 
                        batch_size=32, num_workers=8,
                        rand_seq=True, rand_seq_prob=0.5,
                        excluce_image_mask=False):
    
    dset = IDCardBaseDataset(root=path, tokenizer=tokenizer, mode=mode, 
                             rand_seq=rand_seq, rand_seq_prob=rand_seq_prob,
                             excluce_image_mask=excluce_image_mask)
    
    dsampler = RandomSampler(dset)
    dloader = DataLoader(dset, sampler=dsampler, 
                          batch_size=batch_size, 
                          num_workers=num_workers,
                          collate_fn=None)
    
    return dloader



def get_base_loader(path, tokenizer, 
                    batch_size=32, num_workers=16, 
                    rand_seq=True, rand_seq_prob=0.5,
                    excluce_image_mask=False):
    
    train_loader = get_base_dataloader(path=path, tokenizer=tokenizer, mode="train",
                                       excluce_image_mask=excluce_image_mask,
                                       batch_size=batch_size, num_workers=num_workers,
                                       rand_seq=rand_seq, rand_seq_prob=rand_seq_prob)
    
        
    valid_loader = get_base_dataloader(path=path, tokenizer=tokenizer, mode="valid",
                                       excluce_image_mask=excluce_image_mask,
                                       batch_size=batch_size, num_workers=num_workers,
                                       rand_seq=rand_seq, rand_seq_prob=rand_seq_prob)
    
    return train_loader, valid_loader
    

def get_dataloader(path, tokenizer, mode="train", 
                   batch_size=32, num_workers=8,
                   rand_seq=True, rand_seq_prob=0.5,
                   excluce_image_mask=False):
    
    dset = IDCardDataset(root=path, tokenizer=tokenizer, mode=mode, 
                         rand_seq=rand_seq, rand_seq_prob=rand_seq_prob,
                         excluce_image_mask=excluce_image_mask)
    
    dsampler = RandomSampler(dset)
    dloader = DataLoader(dset, sampler=dsampler, 
                          batch_size=batch_size, 
                          num_workers=num_workers,
                          collate_fn=None)
    
    return dloader

def get_loader(path, tokenizer, 
               batch_size=32, num_workers=16, 
               rand_seq=True, rand_seq_prob=0.5,
               excluce_image_mask=False):
    
    train_loader = get_dataloader(path=path, tokenizer=tokenizer, mode="train",
                                  batch_size=batch_size, num_workers=num_workers,
                                  excluce_image_mask=excluce_image_mask,
                                  rand_seq=rand_seq, rand_seq_prob=rand_seq_prob)
    
        
    valid_loader = get_dataloader(path=path, tokenizer=tokenizer, mode="valid",
                                  excluce_image_mask=excluce_image_mask,
                                  batch_size=batch_size, num_workers=num_workers,
                                  rand_seq=rand_seq, rand_seq_prob=rand_seq_prob)
    
    return train_loader, valid_loader
    
    

# path = '/data/idcard/combined/1606498320/'
# validset = IDCardDataset(root=path, tokenizer=tokenizer, mode='valid', rand_seq=True)

# train_sampler = RandomSampler(trainset)
# valid_sampler = RandomSampler(validset)

# train_loader = DataLoader(trainset, sampler=train_sampler, 
#                           batch_size=BATCH_SIZE, 
#                           num_workers=8,
#                           collate_fn=None)

# valid_loader = DataLoader(validset, sampler=valid_sampler, 
#                           batch_size=BATCH_SIZE, 
#                           num_workers=8,
#                           collate_fn=None)