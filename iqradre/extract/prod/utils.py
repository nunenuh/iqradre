
import re

import torch
import numpy as np
import pandas as pd
from ..trainer import metrics
from ..config import label as label_cfg
from ..config import token as token_cfg
from collections import OrderedDict
from ..ops import boxes_ops

def build_annoset(text_list, boxes):
    boxes_list = boxes_ops.batch_coord2xymm(boxes, to_int=True).tolist()    
    annoset = [{'text':t, "bbox": b}  for t,b in zip(text_list, boxes_list)] 
    return annoset

def annoset_inputs(data_dict, device):
    input_ids = torch.tensor(data_dict['token_ids'], dtype=torch.long)
    mask = torch.tensor(data_dict['mask'], dtype=torch.long)
    bbox = torch.tensor(data_dict['bboxes'], dtype=torch.long)
    
    input_data = {
        'input_ids': input_ids.unsqueeze(dim=0).to(device),
        'attention_mask': mask.unsqueeze(dim=0).to(device),
        'bbox': bbox.unsqueeze(dim=0).to(device)
    }
    return input_data


def annoset_transform(objects, tokenizer, max_seq_length = 512):
    data_anno = tokenize_duplicate_dict(objects, tokenizer)
    texts, bboxes, tokens, token_ids, wseq, gseq, mask = [],[],[],[],[],[],[]

    texts.append(token_cfg.cls_token)
    bboxes.append(token_cfg.cls_token_box)
    tokens.append(token_cfg.cls_token)
    token_ids.append(token_cfg.cls_token_id)
    wseq.append(token_cfg.ignore_index_token_id)
    gseq.append(token_cfg.ignore_index_token_id)
    mask.append(1)

    for obj in data_anno:
        texts.append(obj['text'])
        bboxes.append(obj['bbox'])
        tokens.append(obj['token'])
        token_ids.append(obj['token_id'])
        wseq.append(obj['wseq'])
        gseq.append(obj['gseq'])
        mask.append(1)
        

    texts.append(token_cfg.sep_token)
    bboxes.append(token_cfg.sep_token_box)
    tokens.append(token_cfg.sep_token)
    token_ids.append(token_cfg.sep_token_id)
    wseq.append(token_cfg.ignore_index_token_id)
    gseq.append(token_cfg.ignore_index_token_id)
    mask.append(1)
    
    
    pad_length = max_seq_length - len(texts)
    for p in range(pad_length):
        texts.append(token_cfg.pad_token)
        bboxes.append(token_cfg.pad_token_box)
        tokens.append(token_cfg.pad_token)
        token_ids.append(token_cfg.pad_token_id)
        wseq.append(token_cfg.ignore_index_token_id)
        gseq.append(token_cfg.ignore_index_token_id)
        mask.append(0)
    
    data_dict = {
        'words':texts,
        'bboxes': bboxes,
        'tokens': tokens,
        'token_ids': token_ids,
        'mask': mask,
        'gseq': gseq,
        'wseq': wseq
    }
    
    return data_dict


def tokenize_duplicate_dict(objects, tokenizer):
    new_objects = []
    gseq = 0
    for idx, obj in enumerate(objects):
        curr_text = objects[idx]['text']
        token = tokenizer.tokenize(curr_text)
        if len(token) > 1:
            wseq = 0
            for tok in token:
                
                new_obj = objects[idx].copy()
                new_obj['token'] = tok
                new_obj['token_id'] = tokenizer.convert_tokens_to_ids(tok)
                new_obj['fraction'] = True
                new_obj['wseq'] = wseq
                new_obj['gseq'] = gseq
                new_objects.append(new_obj)
                wseq+=1
                
            gseq+=1
                
        else:
            if len(token)==0:
                obj['token'] = '[UNK]'
                obj['token_id'] = tokenizer.convert_tokens_to_ids('[UNK]')
            else:
                obj['token'] = token[0]
                obj['token_id'] = tokenizer.convert_tokens_to_ids(token[0])
                
            
            obj['fraction'] = False
            obj['wseq'] = 0
            obj['gseq'] = gseq
            new_objects.append(obj)
            gseq+=1

    return new_objects


def normalized_prediction(outputs, tokenizer):
    preds = prediction_index(outputs)
    
    bsize = preds.shape[0]
    
    labels = []
    for idx in range(bsize):
        label_pred = []
        for pds in preds[idx].tolist():
            lbl = label_cfg.idx_to_label.get(pds, "O")
            label_pred.append(lbl)
        labels.append(label_pred)
    
    return labels

    
def prediction_index(outputs):
    if len(outputs)>1:
        preds = outputs[1]
    else:
        preds = outputs[0]
    preds = torch.argmax(preds, dim=2)
    return preds


def clean_prediction_data(data_dict, tokenizer):
    words = data_dict['words']
    boxes = data_dict['bboxes']
    tokens = data_dict['tokens']
    labels = data_dict['labels']
    gseq = data_dict['gseq']
    wseq = data_dict['wseq']

    data = {
        'words':[],
        'bboxes': [],
        'tokens': [],
        'labels': [],
        'gseq': [],
        'wseq': [],
    }

    for (w,b,t,l,gq,wq) in zip(words, boxes, tokens, labels, gseq, wseq):
        if not (w==tokenizer.cls_token or 
                w==tokenizer.sep_token or 
                w==tokenizer.pad_token):

            data['words'].append(w)
            data['bboxes'].append(b)
            data['tokens'].append(t)
            data['labels'].append(l)
            data['gseq'].append(gq)
            data['wseq'].append(wq)
            
    return data

def sort_multidim(data, ygap=28):
    # x[1] sort by BILOU
    sorter = lambda x: (x[1])
    data = sorted(data, key=sorter)
#     print(data)
    
    # x[2][1] sort by x position
    sorter = lambda x: (x[2][1])
    data = sorted(data, key=sorter)
    
    bx = np.array([val[2] for idx, val in enumerate(data)])
    if len(bx)>0:
        order, sub_order = [], []
        for idx, val in enumerate(data):
            if idx == 0: sub_order.append(val)
            if idx > 0:
                box_prev, box_now = data[idx-1][2], data[idx][2]
                box_ygap = box_now[1] - box_prev[1]
                if box_ygap<ygap:
                    sub_order.append(val)
                else:
                    order.append(sub_order)
                    sub_order = []
                    sub_order.append(val)
                    
                    
            if idx == len(data)-1:
                if len(sub_order)==0:
                    sub_order.append(val)
                order.append(sub_order)
                
        reorder = []
        for dt in order:          
            dt = sorted(dt, key=lambda x: x[2][0])
            reorder+=dt
            
        data = reorder
        
    
    return data



def word_taken(data):
    str_out = ""
    for idx in range(len(data)):
        w = data[idx][0]
        if w!="" and len(w)!=0:
            str_out += w
            if idx!=len(data)-1:
                str_out += " "
            
    return str_out


def rebuild_prediction_data(data):
    df = pd.DataFrame(data)
    dfg = df.groupby('gseq').aggregate({
        'words': 'min', 
        'bboxes':'last',
        'tokens':'sum',
        'labels':'first'
    })
    
    base_data = dict((k,[]) for k,v in label_cfg.base_label_name.items())
    for idx in range(len(dfg)):
        labels = dfg.iloc[idx]['labels']
        bbox = dfg.iloc[idx]['bboxes']
        if not labels=="O":
            bil, val = labels.split("-")
            val_type, val_label = val.split("_")
            if val_type=="FLD":
                word = dfg.iloc[idx]['words']
                if "PROVINSI" in word or "KOTA" in word or "KABUPATEN" in word:
                    key = label_cfg.label_to_name[val_label]
                    word = re.sub('[^A-Za-z0-9]+', '', word)
                    base_data[key].append((word, bil, bbox))
                
                
            if val_type=="VAL":
                word = dfg.iloc[idx]['words']
                key = label_cfg.label_to_name[val_label]
                base_data[key].append((word, bil, bbox))


    for k,v in base_data.items():
#         print(v)
        sorted_data = sort_multidim(v)
        base_data[k] = word_taken(sorted_data)
    
    return base_data


def tanggal_lahir_splitting(data):
    ttl = data['ttl']
    tempat, tgl = '', ''
    if ',' in ttl:
        ttl_split = ttl.split(',')
        if len(ttl_split)==2:
            (tempat, tgl) = ttl_split
            tempat, tgl = tempat.strip(), tgl.strip()

    return tempat, tgl


def post_process(data):
#     data_pred = data['prediction']
    data_pred = data
    tempat, tgl = tanggal_lahir_splitting(data)
    pred = OrderedDict()
    pred['provinsi'] = data_pred['provinsi']
    pred['kabupaten'] = data_pred['kabupaten']
    pred['nik'] = data_pred['nik']
    pred['nama'] = data_pred['nama']
    pred['ttl'] = data_pred['ttl']
    pred['tempat_lahir'] = tempat
    pred['tanggal_lahir'] = tgl
    pred['gender'] = data_pred['gender']
    pred['goldar'] = data_pred['goldar']
    pred['alamat'] = data_pred['alamat']
    pred['rtrw'] = data_pred['rtrw']
    pred['kelurahan'] = data_pred['kelurahan']
    pred['kecamatan'] = data_pred['kecamatan']
    pred['agama'] = data_pred['agama']
    pred['perkawinan'] = data_pred['perkawinan']
    pred['pekerjaan'] = data_pred['pekerjaan']
    pred['kewarganegaraan'] = data_pred['kewarganegaraan']
    pred['berlaku'] = data_pred['berlaku']
    pred['sign_place'] = data_pred['sign_place']
    pred['sign_date'] = data_pred['sign_date']
    
#     data['prediction'] = pred
    return data
    