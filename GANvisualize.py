import torch
import torchvision
from model import NetG
from GAN import Config
opt=Config()
netg=NetG(opt)
netg.load_state_dict(torch.load('./result/model/netg_379.pt'))
fix_noises=torch.randn(8,opt.nz,1,1)
fix_fake_img=netg(fix_noises)
torchvision.utils.save_image(fix_fake_img.data,'%s%s.png'%('./gansample/','example'),normalize=True,range=(-1,1),nrow=4)
# for i in range(10000):
# 	fix_noises=torch.randn(1,opt.nz,1,1)
# 	fix_fake_img=netg(fix_noises)
# 	torchvision.utils.save_image(fix_fake_img.data,'%s%s.png'%('./gansample/0/',i),normalize=True,range=(-1,1))