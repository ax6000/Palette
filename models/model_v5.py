import torch
import tqdm
from core.base_model import BaseModel
from core.logger import LogTracker
import copy
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import glob
import os
class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Palette(BaseModel):
    def __init__(self, networks, losses, sample_num, task, optimizers, ema_scheduler=None, **kwargs):
        ''' must to init BaseModel with kwargs '''
        super(Palette, self).__init__(**kwargs)

        ''' networks, dataloder, optimizers, losses, etc. '''
        self.loss_fn = losses[0]
        self.netG = networks[0]
        if ema_scheduler is not None:
            self.ema_scheduler = ema_scheduler
            self.netG_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
        else:
            self.ema_scheduler = None
        
        ''' networks can be a list, and must convert by self.set_device function if using multiple GPU. '''
        self.netG = self.set_device(self.netG, distributed=self.opt['distributed'])
        if self.ema_scheduler is not None:
            self.netG_EMA = self.set_device(self.netG_EMA, distributed=self.opt['distributed'])
        self.load_networks()
        
        self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizers[0])
        self.optimizers.append(self.optG)
        self.resume_training() 

        # load vae
        def load_best_epoch(model,path):
            paths = glob.glob(os.path.join(os.path.dirname(path),"[0-9]*.pth"))
            print(paths)
            best_epoch = sorted([int(os.path.splitext(os.path.basename(p))[0]) for p in paths])[-1]
            print('best epoch is epoch',best_epoch)
            model.load_state_dict(torch.load(path.format(t=best_epoch)))
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
        from models.VAE import VanillaVAE
        self.abp_vae = VanillaVAE(in_channels = 1, latent_dim=16,hidden_dims=[32,64,128,256,512]).to('cuda')
        self.ppg_vae = VanillaVAE(in_channels = 1, latent_dim=16,hidden_dims=[32,64,128,256,512])
        load_best_epoch(self.abp_vae,"..\\..\\notebooks\\outputs\\0531_bp\\{t}.pth")
        load_best_epoch(self.ppg_vae,"..\\..\\notebooks\\outputs\\0531\\{t}.pth")
        
        # self.abp_vae.encoder = self.abp_vae.encoder.to('gpu')
        # self.abp_vae.fc_mu = self.abp_vae.fc_mu.to('gpu')
        # self.abp_vae.fc_var = self.abp_vae.fc_var.to('gpu')
        self.ppg_vae.encoder = self.ppg_vae.encoder.to('cuda')
        self.ppg_vae.fc_mu = self.ppg_vae.fc_mu.to('cuda')
        self.ppg_vae.fc_var = self.ppg_vae.fc_var.to('cuda')
        
        if self.opt['distributed']:
            self.netG.module.set_loss(self.loss_fn)
            self.netG.module.set_new_noise_schedule(phase=self.phase)
        else:
            self.netG.set_loss(self.loss_fn)
            self.netG.set_new_noise_schedule(phase=self.phase)

        ''' can rewrite in inherited class for more informations logging '''
        self.train_metrics = LogTracker(*[m.__name__ for m in losses], phase='train')
        self.val_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='val')
        self.test_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='test')

        self.sample_num = sample_num
        self.task = task
        self.scaler = torch.cuda.amp.GradScaler(self.opt['grad_scale'])
    def set_input(self, data):
        ''' must use set_device in tensor '''
        self.cond_image = self.set_device(data.get('cond_image'))
        self.gt_image = self.set_device(data.get('gt_image'))
        self.mask = self.set_device(data.get('mask'))
        self.mask_image = data.get('mask_image')
        self.path = data['path']
        self.batch_size = len(data['path'])
        # print("#####batch_size ###########",self.batch_size)
    
    def get_current_visuals(self, phase='train'):
        dict = {
            # 'gt_image': (self.gt_image.detach()[:].float().cpu()+1)/2,
            # 'cond_image': (self.cond_image.detach()[:].float().cpu()+1)/2,
        }
        # if self.task in ['inpainting','uncropping']:
        #     dict.update({
        #          'gt_image': (self.gt_image.detach()[:].float().cpu()),
        #     'cond_image': (self.cond_image.detach()[:].float().cpu()),
        #         'mask': self.mask.detach()[:].float().cpu(),
        #         'mask_image': (self.mask_image),
        #     })
        # if phase != 'train':
        dict.update({
                'gt_image': (self.gt_image.detach()[:].float().cpu()),
        'cond_image': (self.cond_image.detach()[:].float().cpu()),
            'output': (self.abp_vae.decode(self.output).detach()[:].float().cpu())
        })
        return dict

    def save_current_results(self):
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            ret_path.append('GT_{}'.format(self.path[idx]))
            ret_result.append(self.gt_image[idx].detach().float().cpu())

            ret_path.append('Process_{}'.format(self.path[idx]))
            ret_result.append(self.cond_image[idx::self.batch_size].detach().float().cpu())
            
            ret_path.append('Out_{}'.format(self.path[idx]))
            ret_result.append(self.abp_vae.decode(self.output[idx-self.batch_size]).detach().float().cpu())
        
        if self.task in ['inpainting','uncropping']:
            ret_path.extend(['Mask_{}'.format(name) for name in self.path])
            ret_result.extend(self.mask_image)

        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)
        return self.results_dict._asdict()

    def bfs_grad_fn(self,grad_fn):
        if grad_fn is None:
            return
        print(grad_fn,grad_fn.next_functions)
        for input_tensor in grad_fn.next_functions:
            self.bfs_grad_fn(input_tensor[0])
    def train_step(self):
        self.netG.train()
        self.train_metrics.reset() 
        for train_data in tqdm.tqdm(self.phase_loader):
            self.set_input(train_data)
            self.optG.zero_grad()
            # with torch.autograd.detect_anomaly():
            loss = self.netG(self.abp_vae.reparameterize(*self.abp_vae.encode(self.gt_image)),
                             self.ppg_vae.reparameterize(*self.ppg_vae.encode(self.cond_image)), mask=self.mask)
            loss.backward()
            self.optG.step()
            self.iter += self.batch_size
            self.writer.set_iter(self.epoch, self.iter, phase='train')
            self.train_metrics.update(self.loss_fn.__name__, loss.item())
            if self.iter % self.opt['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))
                    self.writer.add_scalar(key, value)
                # for key, value in self.get_current_visuals().items():
                #     if len(value.shape) == 3:
                #         pass
                #         # self.writer.add_figure(key,self.make_figures(value))
                #     else:
                #         self.writer.add_images(key, value)
            # if self.iter % self.opt['train']['log_iter']*5 == 0:
            #     self.make_figures2()
            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG)
            # if self.iter % 
        for scheduler in self.schedulers:
            scheduler.step()
        return self.train_metrics.result()
    def make_figures(self,value,single=False):
        figs = []
        if single:
            fig = plt.figure()
            plt.plot(value[-1].squeeze())
            return [fig]
        for i in range(value.shape[0]):
            y = value[i].squeeze()
            fig = plt.figure()
            plt.plot(y)
            figs.append(fig)
    def make_figures2(self):
        visuals = self.get_current_visuals(phase='val')
        batch_size = visuals['gt_image'].shape[0]
        figs_abp = []
        figs_ppg = []
        gt = visuals['gt_image'].squeeze()[0]
        out = visuals['output'].squeeze()[0]
        cond = visuals['cond_image'].squeeze()[0]
        fig_abp = plt.figure()
        plt.plot(gt)
        plt.plot(out,color='orange')
        fig_ppg = plt.figure()
        plt.plot(cond)
        self.writer.add_figure('abp',fig_abp,close=True)
        self.writer.add_figure('ppg',fig_ppg,close=True)
    def val_step(self):
        self.netG.eval()
        self.val_metrics.reset()
        with torch.no_grad():
            for val_data in tqdm.tqdm(self.val_loader):
                self.set_input(val_data)
                # with torch.autocast(device_type="cuda",dtype=torch.float16): 
                # if self.opt['distributed']:
                #     if self.task in ['inpainting','uncropping']:
                #         self.output, self.visuals = self.netG.module.restoration(self.cond_image, y_t=self.cond_image, 
                #             y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                #     else:
                #         self.output, self.visuals = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num)
                # else:
                #     if self.task in ['inpainting','uncropping']:
                #         self.output, self.visuals = self.netG.restoration(self.cond_image, y_t=self.cond_image, 
                #             y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                #     else:
                self.output, self.visuals = self.netG.sample(S=100, batch_size=self.batch_size,shape=(1,256),conditioning=self.ppg_vae.reparameterize(*self.ppg_vae.encode(self.cond_image)), sample_num=self.sample_num)
                
                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='val')

                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.val_metrics.update(key, value)
                    # self.writer.add_scalar(key, value)
                # self.make_figures2()
                for key, value in self.get_current_visuals(phase='val').items():
                    # if len(value.shape) == 3:
                        # self.writer.add_figure(key,self.make_figures(value),close=True)
                    # else:
                    self.writer.add_images(key, value)
                self.writer.save_images(self.save_current_results())

        return self.val_metrics.result()

    def test(self):
        self.netG.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            for phase_data in tqdm.tqdm(self.phase_loader):
                self.set_input(phase_data)
                # with torch.autocast(device_type="cuda",dtype=torch.float32): 
                if self.opt['distributed']:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, y_t=self.cond_image, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num)
                else:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, y_t=self.cond_image, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                    else:
                        # self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num)
                        self.output, self.visuals = self.netG.sample(S=200, batch_size=self.batch_size,shape=(1,256),conditioning=self.cond_image, sample_num=self.sample_num)
                        
                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='test')
                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.test_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                if self.iter % 2048 == 0:
                    # self.make_figures2()
                    for key, value in self.get_current_visuals(phase='test').items():
                        self.writer.add_images(key, value)
                        # self.writer.add_figure(key,self.make_figures(value,single=True),close=True)
                self.writer.save_images(self.save_current_results())
        
        test_log = self.test_metrics.result()
        ''' save logged informations into log dict ''' 
        test_log.update({'epoch': self.epoch, 'iters': self.iter})

        ''' print logged informations to the screen and tensorboard ''' 
        for key, value in test_log.items():
            self.logger.info('{:5s}: {}\t'.format(str(key), value))

    def load_networks(self):
        """ save pretrained model and training state, which only do on GPU 0. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.load_network(network=self.netG, network_label=netG_label, strict=False)
        if self.ema_scheduler is not None:
            self.load_network(network=self.netG_EMA, network_label=netG_label+'_ema', strict=False)

    def save_everything(self):
        """ load pretrained model and training state. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.save_network(network=self.netG, network_label=netG_label)
        if self.ema_scheduler is not None:
            self.save_network(network=self.netG_EMA, network_label=netG_label+'_ema')
        self.save_training_state()

    def distill(self):
        while self.epoch <= self.opt['train']['n_epoch'] and self.iter <= self.opt['train']['n_iter']:
            self.epoch += 1
            if self.opt['distributed']:
                ''' sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas use a different random ordering for each epoch '''
                self.phase_loader.sampler.set_epoch(self.epoch) 

            train_log = self.train_step()

            ''' save logged informations into log dict ''' 
            train_log.update({'epoch': self.epoch, 'iters': self.iter})

            ''' print logged informations to the screen and tensorboard ''' 
            for key, value in train_log.items():
                self.logger.info('{:5s}: {}\t'.format(str(key), value))
            
            if self.epoch % self.opt['train']['save_checkpoint_epoch'] == 0:
                self.logger.info('Saving the self at the end of epoch {:.0f}'.format(self.epoch))
                self.save_everything()

            if self.epoch % self.opt['train']['val_epoch'] == 0:
                self.logger.info("\n\n\n------------------------------Validation Start------------------------------")
                if self.val_loader is None:
                    self.logger.warning('Validation stop where dataloader is None, Skip it.')
                else:
                    val_log = self.val_step()
                    for key, value in val_log.items():
                        self.logger.info('{:5s}: {}\t'.format(str(key), value))
                self.logger.info("\n------------------------------Validation End------------------------------\n\n")
        self.logger.info('Number of Epochs has reached the limit, End.')