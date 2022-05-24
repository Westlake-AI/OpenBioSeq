import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from PIL import Image

from openbioseq.models.utils import Canny, Laplacian, Sobel
from openbioseq.models.utils import PlotTensor
try:
    from skimage.feature import hog, local_binary_pattern
except:
    print("Please install scikit-image.")


def load_image(path, process, img_num=4):
    img_name_list = os.listdir(path)
    assert len(img_name_list) >= img_num
    
    img = list()
    for i in range(img_num):
        img.append(
            process(Image.open(os.path.join(path, img_name_list[i]))).unsqueeze(0)
        )
    return torch.cat(img, dim=0)


def plot_spectrum(im_fft):
    from matplotlib.colors import LogNorm
    # A logarithmic colormap
    plt.figure()
    plt.imshow(np.log(np.abs(im_fft)))
    plt.colorbar()
    

def random_masking(x, mask_ratio=0.9):
    h = x.shape[1]
    w = x.shape[2]

    raw = torch.zeros((h*w,))
    raw[:int(mask_ratio * h * w)] = 1.  # set EXACTLY 30% of the pixels in the mask
    ridx = torch.randperm(h*w)   # a random permutation of the entries
    mask = torch.reshape(raw[ridx], (h, w))

    return mask


def edge_feature_process(x, mode='Sobel', feat_args=dict()):
    feat_args = dict(
        Canny=dict(
            non_max_suppression=True, to_grayscale=False,
            edge_smooth=feat_args.get('edge_smooth', True),
        ),
        Sobel=dict(
            isotropic=True, out_channels=2, to_grayscale=False,
            use_threshold=feat_args.get('use_threshold', False),
        ),
        Laplacian=dict(
            mode=feat_args.get('method', 'DoG'), to_grayscale=False,
            use_threshold=feat_args.get('use_threshold', False)
        ),
    )
    assert mode in feat_args.keys()
    feat_layer = eval(mode)(**feat_args[mode]).cuda()
    if x.dim() == 3:
        x = x.unsqueeze(0)
    assert x.size(1) in [1, 3]

    x = feat_layer(x)
    return x


def texture_feature_process(x, mode='LBP', feat_args=dict()):
    feat_args = dict(
        LBP=dict(
            P=feat_args.get('P', 8),
            R=feat_args.get('R', 8),
            method=feat_args.get('method', 'ror')),
    )
    assert mode in feat_args.keys()

    if mode == 'LBP':
        feat = list()
        B, C, H, W = x.size()
        x = np.array(x.cpu().numpy())
        for i in range(B):
            lbp_feat = local_binary_pattern(x[i, ...].squeeze(), **feat_args[mode]).reshape(1, H, W)
            lbp_feat = lbp_feat / np.max(lbp_feat)
            # lbp_feat = x[i, ...]
            feat.append(lbp_feat)
        feat = np.array(feat)
        feat = torch.from_numpy(feat).type(torch.float32)
    
    return feat.cuda()


def visualize_feature_fft(img, img_fft, feat, feat_fft, mode):
    nrow = img.size(0)
    save_name = "tools/analysis_tools/analysis_features/" + mode + ".png"
    plot_args = dict(dpi=200, apply_inv=False, overwrite=True)

    B, C, H, W = img.size()
    img_fft_full = torch.zeros(img.size()).cuda()
    img_fft_full[:, :, :H // 2, W // 2 - 1:] = img_fft[:, :, H // 2:, :]
    img_fft_full[:, :, H // 2:, W // 2 - 1:] = img_fft[:, :, :H // 2, :]
    img_fft_full[:, :, :H // 2, :W // 2 ] = torch.flip(img_fft[:, :, :H // 2, 1:], dims=(2,3))
    img_fft_full[:, :, H // 2:, :W // 2 ] = torch.flip(img_fft[:, :, H // 2:, 1:], dims=(2,3))
    
    feat_fft_full = torch.zeros(feat.size()).cuda()
    feat_fft_full[:, :, :H // 2, W // 2 - 1:] = feat_fft[:, :, H // 2:, :]
    feat_fft_full[:, :, H // 2:, W // 2 - 1:] = feat_fft[:, :, :H // 2, :]
    feat_fft_full[:, :, :H // 2, :W // 2 ] = torch.flip(feat_fft[:, :, :H // 2, 1:], dims=(2,3))
    feat_fft_full[:, :, H // 2:, :W // 2 ] = torch.flip(feat_fft[:, :, H // 2:, 1:], dims=(2,3))

    img_fft = torch.log(1 + torch.abs(img_fft_full))
    feat_fft = torch.log(1 + torch.abs(feat_fft_full))

    # img_fft = torch.pow(torch.abs(img_fft_full) / 255, 4) * 255 + 0.5
    # feat_fft = torch.pow(torch.abs(feat_fft_full) / 255, 4) * 255 + 0.5

    if feat.size(1) != 3:
        feat = feat.mean(dim=1, keepdim=True)
        # feat = feat.repeat_interleave(3, 1)
    if feat_fft.size(1) != 3:
        feat_fft = feat_fft.mean(dim=1, keepdim=True)
        # feat_fft = feat_fft.repeat_interleave(3, 1)

    img = torch.cat([img, img_fft, feat, feat_fft], dim=0)

    ploter = PlotTensor(apply_inv=False)
    ploter.plot(
        img, nrow=nrow, title_name=mode, save_name=save_name, **plot_args)


preprocess_aug = T.Compose([
   T.Resize((256,256)),
#    T.CenterCrop(224),
   T.ToTensor(),
   T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
   T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
   T.RandomGrayscale(0.2),
])
preprocess_raw = T.Compose([
   T.Resize((256,256)),
   T.Grayscale(num_output_channels=1),
#    T.CenterCrop(224),
   T.ToTensor(),
])
To_tensor = T.Compose([
   T.Resize((256,256)),
   T.ToTensor(),  
])


img = load_image(path="data/CUB200/train/001.Black_footed_Albatross/", process=preprocess_raw, img_num=2).cuda()
img_fft = torch.fft.rfft2(img, dim=(2, 3), norm='ortho')

# feat_mode = 'LBP'
# feat_mode = 'Canny'
# feat_mode = 'Laplacian'
feat_mode = 'Sobel'

if feat_mode in ['Canny', 'Laplacian', 'Sobel']:
    feat = edge_feature_process(
        img, mode=feat_mode, feat_args=dict(method='LoG', edge_smooth=True, use_threshold=True))
else:
    feat = texture_feature_process(
        img, mode=feat_mode, feat_args=dict(method='ror'))

feat_fft = torch.fft.rfft2(feat, dim=(2, 3), norm='ortho')

visualize_feature_fft(img, img_fft, feat=feat, feat_fft=feat_fft, mode=feat_mode)

