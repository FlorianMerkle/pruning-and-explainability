import PIL, torch, torchvision

import numpy as np

def load_images(IMG_PATH, normalize=False):
    img_batch = torch.empty((0,3,224,224))
    for img in sorted(IMG_PATH.iterdir()):
        img = PIL.Image.open(img)
        img = process_imgs(img)
        img_batch = torch.cat((img_batch,img.unsqueeze(0)))
    if normalize:
        img_batch = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img_batch)
    return img_batch

def load_labels(IMG_PATH):
    labels = []
    for i,f in enumerate(list(sorted(IMG_PATH.iterdir()))):
        labels.append(int(f.name[-6]))
    labels = torch.Tensor(labels)
    return labels
    
def load_et_maps(ETM_PATH):
    etm_batch = torch.empty((0,1,224,224))
    for mask in sorted(ETM_PATH.iterdir()):
        etm = process_maps(PIL.Image.open(mask))
        etm_batch = torch.cat((etm_batch,torch.Tensor(etm).unsqueeze(0)))
    return etm_batch


classes = {
    0:'fish',
    1:'dog',
    2:'cassette player',
    3:'chainsaw',
    4:'church',
    5:'music instrument',
    6:'garbage truck',
    7:'gas',
    8:'golfball',
    9:'parachute',
}

def process_imgs(img):
    img = img.resize((224,224))
    x = np.asarray(img)
    if len(x.shape) != 3:
        x = np.expand_dims(x,2)
        x = np.concatenate((x,x,x),2)
    x = np.transpose(x, (2,0,1))/255
    return torch.Tensor(x)

def process_maps(etm):
    etm = etm.resize((224,224))
    x = np.asarray(etm)
    x = np.transpose(x, (2,0,1))[3]/255
    return np.expand_dims(x, axis=0)

def load_imagenette(path, bs=32, normalize=True):
    transforms = [
        torchvision.transforms.Resize(256),
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.ToTensor(),]
    normalization = [torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    if normalize: transforms = torchvision.transforms.Compose(transforms+normalization)
    else: composed_transforms = torchvision.transforms.Compose(transforms)


    train_path= path+'/train'
    imagenette_train = torchvision.datasets.ImageFolder(
        root=train_path,
        transform=composed_transforms
    )
    val_path=path+'/val'
    imagenette_val = torchvision.datasets.ImageFolder(
        root=val_path,
        transform=composed_transforms
    )

    train_loader = torch.utils.data.DataLoader(imagenette_train, num_workers=4,
                                              batch_size=bs,
                                              shuffle=True)
    val_loader = torch.utils.data.DataLoader(imagenette_val, num_workers=4,
                                              batch_size=bs,
                                              shuffle=True)
    return train_loader, val_loader