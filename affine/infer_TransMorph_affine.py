from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses, random, math
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph_affine import CONFIGS as CONFIGS_TM
import models.TransMorph_affine as TransMorph
import torch.nn as nn

def affine_aug(im, im_label=None, seed=10):
    # mode = 'bilinear' or 'nearest'
    with torch.no_grad():
        random.seed(seed)
        angle_range = 10
        trans_range = 0.1
        scale_range = 0.1
        # scale_range = 0.15

        angle_xyz = (random.uniform(-angle_range, angle_range) * math.pi / 180,
                     random.uniform(-angle_range, angle_range) * math.pi / 180,
                     random.uniform(-angle_range, angle_range) * math.pi / 180)
        scale_xyz = (random.uniform(-scale_range, scale_range), random.uniform(-scale_range, scale_range),
                     random.uniform(-scale_range, scale_range))
        trans_xyz = (random.uniform(-trans_range, trans_range), random.uniform(-trans_range, trans_range),
                     random.uniform(-trans_range, trans_range))

        rotation_x = torch.tensor([
            [1., 0, 0, 0],
            [0, math.cos(angle_xyz[0]), -math.sin(angle_xyz[0]), 0],
            [0, math.sin(angle_xyz[0]), math.cos(angle_xyz[0]), 0],
            [0, 0, 0, 1.]
        ], requires_grad=False).unsqueeze(0).cuda()

        rotation_y = torch.tensor([
            [math.cos(angle_xyz[1]), 0, math.sin(angle_xyz[1]), 0],
            [0, 1., 0, 0],
            [-math.sin(angle_xyz[1]), 0, math.cos(angle_xyz[1]), 0],
            [0, 0, 0, 1.]
        ], requires_grad=False).unsqueeze(0).cuda()

        rotation_z = torch.tensor([
            [math.cos(angle_xyz[2]), -math.sin(angle_xyz[2]), 0, 0],
            [math.sin(angle_xyz[2]), math.cos(angle_xyz[2]), 0, 0],
            [0, 0, 1., 0],
            [0, 0, 0, 1.]
        ], requires_grad=False).unsqueeze(0).cuda()

        trans_shear_xyz = torch.tensor([
            [1. + scale_xyz[0], 0, 0, trans_xyz[0]],
            [0, 1. + scale_xyz[1], 0, trans_xyz[1]],
            [0, 0, 1. + scale_xyz[2], trans_xyz[2]],
            [0, 0, 0, 1]
        ], requires_grad=False).unsqueeze(0).cuda()

        theta_final = torch.matmul(rotation_x, rotation_y)
        theta_final = torch.matmul(theta_final, rotation_z)
        theta_final = torch.matmul(theta_final, trans_shear_xyz)

        output_disp_e0_v = F.affine_grid(theta_final[:, 0:3, :], im.shape, align_corners=False)

        im = F.grid_sample(im, output_disp_e0_v, mode='bilinear', padding_mode="border", align_corners=False)

        if im_label is not None:
            im_label = F.grid_sample(im_label, output_disp_e0_v, mode='nearest', padding_mode="border",
                                     align_corners=False)
            return im, im_label
        else:
            return im

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

def main():
    val_dir = '/scratch/jchen/DATA/AutoPET/affine_aligned/network/test/' 
    save_dir = 'TransMorphAffine_cr/'
    lr = 0.0001 # learning rate
    model_dir = 'experiments/'+save_dir
    csv_dir = 'Quantitative_Results/'+save_dir
    if not os.path.exists('Quantitative_Results/'+save_dir):
        os.makedirs('Quantitative_Results/'+save_dir)
    
    dicts = utils.label_names()
    line = 'pat_idx'
    for i in range(40):
        line = line + ',' + dicts[i]
    file_name = 'tmp'
    csv_writter(line+','+'non_jec'+','+'non_jec_vol', csv_dir+file_name)
    
    '''
    Initialize model
    '''
    H, W, D = 192, 192, 256
    num_clus = 40
    config = CONFIGS_TM['TransMorph-3-LVL']
    config.img_size = (H, W, D)
    config.window_size = (H // 32, W // 32, D // 32)
    config.out_chan = 3
    config.use_checkpoint = False
    model = TransMorph.TransMorphAffine(config)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
    model.load_state_dict(best_model)
    model.cuda()
    affine_trans = TransMorph.AffineTransform()#AffineTransformer((H, W, D)).cuda()
    affine_trans_nn = TransMorph.AffineTransform(mode='nearest')

    '''
    Initialize training
    '''
    val_names = glob.glob(val_dir + '*_segsimple*')
    val_names = [name.split('/')[-1].split('_')[0] for name in val_names]
    val_set = datasets.AutoPETTrainDataset(val_dir, val_names)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    
    criterion = losses.LocalCorrRatio()#nn.MSELoss()
    
    best_dsc = 0
        
    '''
    Validation
    '''
    idd = 0
    dsc_raw = []
    eval_dsc = utils.AverageMeter()
    lvls = [16, 8, 4, 2, 1]
    itrs = [100, 100, 120, 140, 160]
    for idx, data in enumerate(val_loader):
        if idx != 4:
            continue
        with torch.no_grad():
            model.eval()
            data = [t.cuda() for t in data]
            x_ct = data[0].cuda().float()
            x_suv = data[1].cuda().float()
            x_seg = data[2].cuda().float()
            x_ct_aff1, x_seg_aff1 = affine_aug(x_ct, x_seg, seed=idx)
            
            x_suv_aff2, x_seg_aff2 = affine_aug(x_suv, x_seg, seed=idx+2434)
        
            aff, scl, transl, shr, aff_, scl_, transl_, shr_ = model((x_suv_aff2, x_ct_aff1))
            x_suv_trans, mat, inv_mat = affine_trans(x_suv_aff2, aff, scl, transl, shr)
            x_seg_trans = affine_trans_nn.apply_affine(x_seg_aff2.float(), mat)
            
        aff_.requires_grad_(); scl_.requires_grad_(); transl_.requires_grad_(); shr_.requires_grad_()
        params = [{'params': aff_, 'lr': lr}] + [{'params': scl_, 'lr': lr}] + \
            [{'params': transl_, 'lr': lr}] + [{'params': shr_, 'lr': lr}]
        optimizer = optim.Adam(params, lr=0.004)
        for i, lvl in enumerate(lvls):
            x = F.avg_pool3d(x_suv_aff2.detach(), lvl)
            y = F.avg_pool3d(x_ct_aff1, lvl).clone()
            for _ in range(itrs[i]):
                transl = transl_.clone()
                aff = torch.clamp(aff_, min=-1, max=1) * np.pi
                scl = scl_.clone() + 1
                scl = torch.clamp(scl, min=0, max=5)
                shr = torch.clamp(shr_, min=-1, max=1) * np.pi
                x_trans, mat, inv_mat = affine_trans(x, aff, scl, transl, shr)
                loss = criterion(x_trans, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(loss)
        x_suv_trans = affine_trans.apply_affine(x_suv_aff2.float(), mat)
        x_seg_trans = affine_trans_nn.apply_affine(x_seg_aff2.float(), mat)
        plt.figure(dpi=250)
        plt.subplot(2, 3, 1)
        plt.imshow(x_ct_aff1.cpu().detach().numpy()[0, 0, :, 96, ], cmap='gray')
        plt.subplot(2, 3, 2)
        plt.imshow(x_suv_aff2.cpu().detach().numpy()[0, 0, :, 96, ], cmap='hot')
        plt.subplot(2, 3, 3)
        plt.imshow(x_ct_aff1.cpu().detach().numpy()[0, 0, :, 96, ], cmap='gray')
        plt.imshow(x_suv_trans.cpu().detach().numpy()[0, 0, :, 96, ], cmap='hot', alpha=0.4)
        plt.subplot(2, 3, 4)
        plt.imshow(x_seg_aff1.cpu().detach().numpy()[0, 0, :, 96, ])
        plt.subplot(2, 3, 5)
        plt.imshow(x_seg_aff2.cpu().detach().numpy()[0, 0, :, 96, ])
        plt.subplot(2, 3, 6)
        plt.imshow(torch.abs(x_seg_trans-x_seg_aff1).cpu().detach().numpy()[0, 0, :, 96, ], cmap='magma')
        plt.savefig('reg_results')
        plt.close()
        sys.exit()
        dsc_line = utils.dice_val_substruct(x_seg_trans.long(), x_seg_aff1.long(), idx)
        csv_writter(dsc_line, csv_dir+file_name)
        
        dsc = utils.dice_val_VOI((x_seg_trans).long(), x_seg_aff1.long(), num_clus)
        eval_dsc.update(dsc.item(), x_ct.size(0))
        print(eval_dsc.avg)
        
def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[0], grid_step):
        grid_img[j+line_thickness-1, :, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=4):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

def seedBasic(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def seedTorch(seed=2021):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()