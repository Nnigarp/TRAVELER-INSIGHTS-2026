"""Staged CPU BETO training and safe V2 artifacts; V1 implementation stays frozen."""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

from .beto_classifier import BetoClassifier,BetoEncoder,softmax
from .beto_encoder_probe import MODEL,REVISION,WEIGHT_SHA256,checkpoint,file_sha
from .contextual_benchmark import selection_score
from .corpus import CorpusError
from .labels import LABELS
from .synthetic_diversity import ignored_target,write_json

SCHEMA='traveler-beto-supervised-v2'


class EncoderV2(BetoEncoder):
    def __init__(self,*args,pooling='mean_l2',**kwargs):
        if pooling not in ('mean_l2','cls_l2'):raise CorpusError('Pooling no soportado.')
        super().__init__(*args,**kwargs);self.pooling=pooling

    def pooled(self,tokens):
        if self.pooling=='mean_l2':return super().pooled(tokens)
        import torch
        hidden=self.model(**tokens).last_hidden_state[:,0]
        return torch.nn.functional.normalize(hidden,p=2,dim=1)


def target_ids(rows):return np.asarray([LABELS.index(r['intended_label']) for r in rows],dtype=np.int64)


def tuning_score(rows,probabilities):
    y=target_ids(rows);result=selection_score(y,probabilities)
    guessed=np.asarray(LABELS)[probabilities.argmax(axis=1)]
    result['negative_to_positive']=int(sum(r['intended_label']=='negativo' and p=='positivo' for r,p in zip(rows,guessed)))
    return result


def score_rank(value):return value['macro_f1'],-value['negative_to_positive'],-value['cross_entropy']


def fit_head(features,rows,tune_features,tune,protocol,seed,*,fixed_epoch=None):
    if any(r['role']!='train' for r in rows) or any(r['role']!='tune' for r in tune):raise CorpusError('Cabeza: entrenamiento/tune independientes requeridos.')
    import torch
    torch.manual_seed(seed);torch.set_num_threads(protocol['threads']);torch.use_deterministic_algorithms(True)
    config=protocol['head'];linear=torch.nn.Linear(768,3);drop=torch.nn.Dropout(config['dropout'])
    optimizer=torch.optim.AdamW(linear.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    x=torch.as_tensor(features,dtype=torch.float32);y=torch.as_tensor(target_ids(rows))
    random=torch.Generator().manual_seed(seed);best=None;weights=None;history=[]
    for epoch in range(1,(fixed_epoch or config['epochs'])+1):
        total=0.;drop.train()
        for ids in torch.randperm(len(rows),generator=random).split(config['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(linear(drop(x[ids])),y[ids]);loss.backward();optimizer.step()
            total+=float(loss.detach())*len(ids)
        record={'epoch':epoch,'train_cross_entropy':total/len(rows)}
        if epoch in config['selection_epochs'] or epoch==fixed_epoch:
            w,b=linear.weight.detach().numpy().copy(),linear.bias.detach().numpy().copy()
            record['tune']=tuning_score(tune,softmax(tune_features@w.T+b))
            if fixed_epoch or best is None or score_rank(record['tune'])>score_rank(best['tune']):
                best,weights=dict(record),(w,b)
        history.append(record)
    return weights,{'selected':best,'history':history,'seed':seed,'encoder_trainable_parameters':0,'head_trainable_parameters':2307}


def _allowed_keys(model,mode):
    if mode=='frozen':return set()
    if mode=='full':return set(model.state_dict())
    if mode=='top2':return {k for k in model.state_dict() if k.startswith(('encoder.layer.10.','encoder.layer.11.'))}
    raise CorpusError('Modo de entrenamiento desconocido.')


def train_encoder(encoder,rows,tune,weights,config,protocol,seed,state_path,*,fixed_epoch=None):
    if any(r['role']!='train' for r in rows) or any(r['role']!='tune' for r in tune):raise CorpusError('Encoder: entrenamiento/tune independientes requeridos.')
    import torch
    from safetensors.torch import save_file,load_file
    from transformers import get_linear_schedule_with_warmup
    torch.manual_seed(seed);torch.set_num_threads(protocol['threads']);torch.use_deterministic_algorithms(True)
    allowed=_allowed_keys(encoder.model,config['mode'])
    for name,param in encoder.model.named_parameters():param.requires_grad_(name in allowed)
    trainable=[p for p in encoder.model.parameters() if p.requires_grad]
    if not trainable:raise CorpusError('Fine-tuning sin parámetros de encoder.')
    linear=torch.nn.Linear(768,3);drop=torch.nn.Dropout(config['head_dropout'])
    with torch.no_grad():linear.weight.copy_(torch.from_numpy(weights[0]));linear.bias.copy_(torch.from_numpy(weights[1]))
    optimizer=torch.optim.AdamW([{'params':trainable,'lr':config['encoder_lr']},
                                {'params':linear.parameters(),'lr':config['head_lr']}],weight_decay=config['weight_decay'])
    epochs=fixed_epoch or config['epochs'];steps=config['epochs']*math.ceil(len(rows)/protocol['finetune_batch_size'])
    scheduler=get_linear_schedule_with_warmup(optimizer,int(steps*protocol['warmup_fraction']),steps)
    random=torch.Generator().manual_seed(seed);y=torch.as_tensor(target_ids(rows));best=None;history=[]
    first=next(p for p in encoder.model.parameters() if p.requires_grad);before=first.detach().clone()
    selected_weights=None;first_gradient=None;started=time.perf_counter()
    state_path.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,epochs+1):
        encoder.model.eval()
        if config['mode']=='full':encoder.model.train()
        else:
            for i in (10,11):encoder.model.encoder.layer[i].train()
        drop.train();total=0.;gradient_norms=[]
        for ids in torch.randperm(len(rows),generator=random).split(protocol['finetune_batch_size']):
            optimizer.zero_grad(set_to_none=True)
            features=encoder.pooled(encoder.tokens([rows[i]['text'] for i in ids.tolist()]))
            loss=torch.nn.functional.cross_entropy(linear(drop(features)),y[ids],label_smoothing=config['label_smoothing'])
            loss.backward()
            if first_gradient is None:first_gradient=float(torch.linalg.vector_norm(first.grad.detach()))
            norm=torch.nn.utils.clip_grad_norm_(trainable+list(linear.parameters()),protocol['gradient_clip_norm'])
            gradient_norms.append(float(norm));optimizer.step();scheduler.step();total+=float(loss.detach())*len(ids)
        encoder.model.eval();drop.eval()
        tune_features=encoder.encode([r['text'] for r in tune])
        w,b=linear.weight.detach().numpy().copy(),linear.bias.detach().numpy().copy()
        record={'epoch':epoch,'train_cross_entropy_smoothed_if_configured':total/len(rows),
                'tune':tuning_score(tune,softmax(tune_features@w.T+b)),
                'gradient_norm_mean_before_clip':float(np.mean(gradient_norms)),
                'last_encoder_lr':optimizer.param_groups[0]['lr']}
        history.append(record)
        if fixed_epoch or best is None or score_rank(record['tune'])>score_rank(best['tune']):
            best,selected_weights=dict(record),(w,b)
            save_file({k:v.detach().cpu().contiguous() for k,v in encoder.model.state_dict().items() if k in allowed},str(state_path))
    encoder.model.load_state_dict(load_file(str(state_path)),strict=False)
    update=float(torch.linalg.vector_norm(first.detach()-before))
    if not np.isfinite(update) or update<=0:raise CorpusError('No se verificó actualización real del encoder.')
    frozen_gradients_absent=all(p.grad is None for name,p in encoder.model.named_parameters() if name not in allowed)
    if not frozen_gradients_absent:raise CorpusError('Gradiente fuera de bloques autorizados.')
    for p in encoder.model.parameters():p.requires_grad_(False)
    return selected_weights,{'selected':best,'history':history,'seed':seed,'fit_wall_seconds':time.perf_counter()-started,
         'encoder_trainable_parameters':sum(p.numel() for p in trainable),'head_trainable_parameters':2307,
         'first_encoder_gradient_norm':first_gradient,'probe_tensor_l2_update':update,
         'frozen_parameter_gradients_absent':frozen_gradients_absent,'adaptation_sha256':file_sha(state_path)}


def export_model(destination,cache,weights,config,training,*,state=None):
    destination=ignored_target(Path(destination),kinds=('artifacts','data/private'))
    if destination.exists():raise CorpusError('Artefacto ya existente.')
    verified=checkpoint(cache);destination.mkdir(parents=True);folder=destination/'checkpoint';folder.mkdir()
    for name in [*verified['files'],'verified_checkpoint.json']:shutil.copy2(cache/name,folder/name)
    np.savez(destination/'head.npz',weight=np.asarray(weights[0],np.float32),bias=np.asarray(weights[1],np.float32))
    metadata={'schema':SCHEMA,'model':MODEL,'revision':REVISION,'base_weight_sha256':WEIGHT_SHA256,
        'checkpoint_manifest':verified,'classes':list(LABELS),'max_length':config['max_length'],'pooling':config['pooling'],
        'mode':config['mode'],'head_sha256':file_sha(destination/'head.npz'),'training':training,
        'human_validated':False,'calibrated_or_policy_validated_in_app':False}
    if state:
        shutil.copy2(state,destination/'adaptation.safetensors');metadata['adaptation_sha256']=file_sha(destination/'adaptation.safetensors')
    write_json(destination/'model_manifest.json',metadata)
    return metadata


def load_model_v2(artifact,*,threads=2):
    artifact=ignored_target(Path(artifact),kinds=('artifacts','data/private'))
    metadata=json.loads((artifact/'model_manifest.json').read_text(encoding='utf-8'))
    if (metadata.get('schema')!=SCHEMA or metadata.get('model')!=MODEL or metadata.get('revision')!=REVISION
            or metadata.get('base_weight_sha256')!=WEIGHT_SHA256 or metadata.get('classes')!=list(LABELS)
            or metadata.get('human_validated') is not False or metadata.get('training',{}).get('supervised') is not True
            or metadata.get('max_length') not in (128,256) or metadata.get('mode') not in ('frozen','top2','full')):
        raise CorpusError('Artefacto V2 incompatible o no supervisado.')
    for name in ('head.npz','model_manifest.json','checkpoint'):
        if not (artifact/name).resolve().is_relative_to(artifact):raise CorpusError('Escape del artefacto.')
    if file_sha(artifact/'head.npz')!=metadata['head_sha256']:raise CorpusError('Cabeza modificada.')
    encoder=EncoderV2(artifact/'checkpoint',pooling=metadata['pooling'],threads=threads,max_length=metadata['max_length'])
    if encoder.manifest!=metadata['checkpoint_manifest']:raise CorpusError('Checkpoint modificado.')
    allowed=_allowed_keys(encoder.model,metadata['mode'])
    if allowed:
        state=artifact/'adaptation.safetensors'
        if not state.resolve().is_relative_to(artifact) or file_sha(state)!=metadata.get('adaptation_sha256'):
            raise CorpusError('Adaptación modificada.')
        from safetensors.torch import load_file
        tensors=load_file(str(state),device='cpu')
        if set(tensors)!=allowed:raise CorpusError('Adaptación fuera del alcance declarado.')
        encoder.model.load_state_dict(tensors,strict=False)
    elif 'adaptation_sha256' in metadata:raise CorpusError('Encoder congelado con adaptación inesperada.')
    with np.load(artifact/'head.npz',allow_pickle=False) as head:
        if set(head.files)!={'weight','bias'}:raise CorpusError('Cabeza no canónica.')
        return BetoClassifier(encoder,head['weight'],head['bias'],metadata)
