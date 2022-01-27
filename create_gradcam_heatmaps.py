import argparse
import torch
import torch.nn as nn
from pathlib import Path

from load_models import load_mobilenet, load_resnet, load_vgg
import cv2
import numpy as np

if torch.cuda.is_available() == True:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(device)
dtype = torch.float32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_architecture', type=str, help='should be one of "VGG16", "ResNet18", or "MobileNet"' )
    args = parser.parse_args()
    model_architecture = args.model_architecture
    assert model_architecture in ['VGG16', 'ResNet18', 'MobileNet']

    if model_architecture == 'VGG16': 
        CPRS = ['CPR1', 'CPR2', 'CPR4', 'CPR8', 'CPR16', 'CPR32', 'CPR64']
        load_model = load_vgg
    if model_architecture == 'ResNet18': 
        CPRS = ['CPR1', 'CPR2', 'CPR4', 'CPR8', 'CPR16']
        load_model = load_resnet
    if model_architecture == 'MobileNet': 
        CPRS = ['CPR1', 'CPR2', 'CPR4', 'CPR8', 'CPR16']
        load_model = load_mobilenet
    IMGS_PATH = Path('./exp-prune-data/imagelist.pt')
    LABELS_PATH = Path('./exp-prune-data/labellist.pt')   
    imgs = torch.stack(torch.load(IMGS_PATH)).to(device)
    labels = torch.stack(torch.load(LABELS_PATH)).to(device)
    bs = 20
    for CPR in CPRS:
        LOAD_PATH = Path(f'./models/{CPR}-{model_architecture}.pt')
        model = load_model(LOAD_PATH, pruned=False if CPR=='CPR1' else True).to(device)
        if model_architecture == 'VGG16': conv_layer = model.features
        if model_architecture == 'ResNet18': conv_layer = model[1].layer4
        if model_architecture == 'MobileNet': conv_layer = model.features

        low_res_hms = []
        upsampeled_hms =[]
        for i in range(int(len(imgs)/bs)):
            img_batch = imgs[i*bs:(i+1)*bs]
            label_batch = labels[i*bs:(i+1)*bs]
            upsampeled_hm, hm = bw_get_gradcam_heatmap(model, img_batch, label_batch, conv_layer)
            low_res_hms.append(hm)
            upsampeled_hms.append(upsampeled_hm)
        low_res_hms = torch.cat(low_res_hms)
        upsampeled_hms = torch.cat(upsampeled_hms)
        #save low-res heatmaps
        SAVE_PATH = './gradcam_hms/low_res/'
        Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
        LR_SAVE_PATH = Path(SAVE_PATH, f'{CPR}-{model_architecture}-lr_heatmaps.pt')
        torch.save(low_res_hms, LR_SAVE_PATH)
        # save high-res heatmaps
        SAVE_PATH = './gradcam_hms/high_res/'
        Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
        HR_SAVE_PATH = Path(f'./gradcam_hms/high_res/{CPR}-{model_architecture}-hr_heatmaps.pt')
        torch.save(upsampeled_hms, HR_SAVE_PATH)

        



    
    



def bw_get_gradcam_heatmap(model, images, labels, conv_layer):
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output
        return hook
    conv_layer.register_forward_hook(get_activation('target_conv_layer'))
    preds = model(images)
    pred_index = preds.argmax(-1)
    activations = activation['target_conv_layer']
    activations.requires_grad_(True)
    activations.retain_grad()
    best_preds=torch.Tensor((preds.shape[0]))
    for i in range(len(preds)):
        #best_preds[i] = preds[i, labels[i]]
        best_preds[i] = preds[i, pred_index[i]]
    best_preds.sum().backward()
    grads = activations.grad
    pooled_grads = grads.mean(axis=(-1,-2))
    for i in range(activations.shape[0]):
        activations[i,:,:,:] = (activations[i].T * pooled_grads[i]).T
    hm = activations.mean(dim=1).squeeze()
    hm = torch.nn.functional.relu_(hm)
    hm = hm/hm.max()
    hm = hm.detach().cpu()
    upsampeled_hm=[]
    for i in range(hm.shape[0]):
        upsampeled_hm.append(cv2.resize(hm[i].numpy(), (images.shape[-1], images.shape[-1])))
    return torch.Tensor(upsampeled_hm), hm


if __name__ == "__main__":
    main()