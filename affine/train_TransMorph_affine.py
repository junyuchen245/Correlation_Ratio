from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses, random, math
import sys
from torch.utils.data import DataLoader
from data import datasets
import numpy as np
import torch
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

def main():
    batch_size = 1
    train_dir = '/scratch/jchen/DATA/AutoPET/affine_aligned/network/train/'
    val_dir = '/scratch/jchen/DATA/AutoPET/affine_aligned/network/val/' 
    save_dir = 'TransMorphAffine_cr/'
    if not os.path.exists('experiments/'+save_dir):
        os.makedirs('experiments/'+save_dir)
    if not os.path.exists('logs/'+save_dir):
        os.makedirs('logs/'+save_dir)
    sys.stdout = Logger('logs/'+save_dir)
    lr = 0.1 # learning rate
    epoch_start = 0
    max_epoch = 500 #max traning epoch
    cont_training = False #if continue training

    '''
    Initialize model
    '''
    H, W, D = 192, 192, 256
    num_clus = 29
    config = CONFIGS_TM['TransMorph-3-LVL']
    config.img_size = (H, W, D)
    config.window_size = (H // 32, W // 32, D // 32)
    config.out_chan = 3
    config.use_checkpoint = False
    model = TransMorph.TransMorphAffine(config)
    model.cuda()
    affine_trans = TransMorph.AffineTransform()#AffineTransformer((H, W, D)).cuda()
    affine_trans_nn = TransMorph.AffineTransform(mode='nearest')
    
    '''
    If continue from previous training
    '''
    if cont_training:
        epoch_start = 201
        model_dir = 'experiments/'+save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
        print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-1]))
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    train_names = glob.glob(train_dir + '*_segsimple*')
    train_names = [name.split('/')[-1].split('_')[0] for name in train_names]
    val_names = glob.glob(val_dir + '*_segsimple*')
    val_names = [name.split('/')[-1].split('_')[0] for name in val_names]
    train_set = datasets.AutoPETTrainDataset(train_dir, train_names)
    val_set = datasets.AutoPETTrainDataset(val_dir, val_names)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    optimizer = optim.Adam(model.parameters(), lr=0.00002)
    criterion = losses.CorrRatio()#losses.MI.loss#losses.CorrRatio()#nn.MSELoss()
    
    best_dsc = 0
    writer = SummaryWriter(log_dir='logs/'+save_dir)
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        idx = 0
        loss_all = utils.AverageMeter()
        for data in train_loader:
            idx += 1
            model.train()
            rdn_int1 = random.randint(1000, 10000)
            rdn_int2 = random.randint(1000, 10000)
            with torch.no_grad():
                x_ct = data[0].cuda().float()
                x_suv = data[1].cuda().float()
                x_seg = data[2].cuda().float()
                x_ct_aff1, x_seg_aff1 = affine_aug(x_ct, x_seg, seed=rdn_int1)
                
                x_suv_aff2, x_seg_aff2 = affine_aug(x_suv, x_seg, seed=rdn_int2)
            
            aff, scl, transl, shr = model((x_suv_aff2, x_ct_aff1))
            x_suv_trans, mat, inv_mat = affine_trans(x_suv_aff2, aff, scl, transl, shr)
            x_ct_trans = affine_trans.apply_affine(x_ct_aff1, inv_mat)
            x_seg_trans = affine_trans_nn.apply_affine(x_seg_aff2.float(), mat)
            
            
            #print(mat)
            loss = criterion(x_suv_trans, x_ct_aff1)/2 + criterion(x_ct_trans, x_suv_aff2)/2
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            '''
            plt.figure()
            plt.subplot(2, 3, 1)
            plt.imshow(x_ct_aff1.cpu().detach().numpy()[0, 0, :, 96, ], cmap='gray')
            plt.subplot(2, 3, 2)
            plt.imshow(x_suv_aff2.cpu().detach().numpy()[0, 0, :, 96, ], cmap='gray_r')
            plt.subplot(2, 3, 3)
            plt.imshow(x_suv_trans.cpu().detach().numpy()[0, 0, :, 96, ], cmap='gray_r')
            plt.subplot(2, 3, 4)
            plt.imshow(x_seg_aff1.cpu().detach().numpy()[0, 0, :, 96, ])
            plt.subplot(2, 3, 5)
            plt.imshow(x_seg_aff2.cpu().detach().numpy()[0, 0, :, 96, ])
            plt.subplot(2, 3, 6)
            plt.imshow(x_seg_trans.cpu().detach().numpy()[0, 0, :, 96, ])
            plt.savefig('reg_results')
            plt.close()
            '''
            loss_all.update(loss.item(), x_ct_aff1.numel())

            print('Iter {} of {} loss {:.4f}'.format(idx, len(train_loader), loss.item()))
        
        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        
        '''
        Validation
        '''
        idd = 0
        dsc_raw = []
        eval_dsc = utils.AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x_ct = data[0].cuda().float()
                x_suv = data[1].cuda().float()
                x_seg = data[2].cuda().float()
                x_ct_aff1, x_seg_aff1 = affine_aug(x_ct, x_seg, seed=idd)
                
                x_suv_aff2, x_seg_aff2 = affine_aug(x_suv, x_seg, seed=idd+123)
                
                aff, scl, transl, shr = model((x_suv_aff2, x_ct_aff1))
                x_suv_trans, mat, inv_mat = affine_trans(x_suv_aff2, aff, scl, transl, shr)
                x_seg_trans = affine_trans_nn.apply_affine(x_seg_aff2.float(), mat)
                
                plt.figure()
                plt.subplot(2, 3, 1)
                plt.imshow(x_ct_aff1.cpu().detach().numpy()[0, 0, :, 96, ], cmap='gray')
                plt.subplot(2, 3, 2)
                plt.imshow(x_suv_aff2.cpu().detach().numpy()[0, 0, :, 96, ], cmap='gray_r')
                plt.subplot(2, 3, 3)
                plt.imshow(x_suv_trans.cpu().detach().numpy()[0, 0, :, 96, ], cmap='gray_r')
                plt.subplot(2, 3, 4)
                plt.imshow(x_seg_aff1.cpu().detach().numpy()[0, 0, :, 96, ])
                plt.subplot(2, 3, 5)
                plt.imshow(x_seg_aff2.cpu().detach().numpy()[0, 0, :, 96, ])
                plt.subplot(2, 3, 6)
                plt.imshow(x_seg_trans.cpu().detach().numpy()[0, 0, :, 96, ])
                plt.savefig('reg_results')
                plt.close()
                idd += 1
                dsc = utils.dice_val_VOI((x_seg_trans).long(), x_seg_aff1.long(), num_clus)
                if epoch == 0:
                    dsc_raw.append(utils.dice_val_VOI(x_seg_trans.long(), x_seg_aff1.long()).item())
                eval_dsc.update(dsc.item(), x_ct.size(0))
                print(eval_dsc.avg)
        if epoch == 0:
            print('raw dice: {}'.format(np.mean(dsc_raw)))
        best_dsc = max(eval_dsc.avg, best_dsc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
        }, save_dir='experiments/'+save_dir, filename='dsc{:.4f}.pth.tar'.format(eval_dsc.avg))
        writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
        plt.switch_backend('agg')
        ct_fig = comput_fig(x_ct_aff1)
        suv_fig = comput_fig(x_suv_aff2)
        suv_aff_fig = comput_fig(x_suv_trans)
        
        writer.add_figure('suv_aff', suv_aff_fig, epoch)
        plt.close(suv_aff_fig)
        writer.add_figure('suv', suv_fig, epoch)
        plt.close(suv_fig)
        writer.add_figure('ct', ct_fig, epoch)
        plt.close(ct_fig)
        loss_all.reset()
    writer.close()

def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, :, 88:96, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[1]):
        plt.subplot(2, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[:, i, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

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