from vis_utils.visualize import reverse_normalize
from vis_utils.cam_functions import visual_batch
import torch.nn.functional as F
import torch


def guiyihua( x):
    x_min = x.min()
    x_max = x.max()
    x_1 = (x - x_min) / (x_max - x_min)
    return x_1

def get_raw_images(images):
    b, s, c, h, w = images.size()
    image = images.view(b * s, c, h, w)
    imgs_vis = []
    for k in range(b * s):
        img = image[k].unsqueeze(0)  # torch.Size([1, 3, 256, 128])
        img = reverse_normalize(img)
        imgs_vis.append(img)
    return imgs_vis

def keshihua(cam, imgs, i, savedir):
    visual_batch(cam, imgs, i, savedir)
    return print('done')

def visualization_sequence(inputs, atte_mask, simi_map, batch, part_num=2):
    b = atte_mask.size(0)//8
    t = 8
    h = 16
    w = 8
    n = part_num
    raw_imgs = get_raw_images(inputs)

    for i in range(n):
        mask = atte_mask[:,i,:,:].reshape(b,t,h*w)
        mask = guiyihua(mask).reshape(b, t, h, w)
        keshihua(mask, raw_imgs, 'part_{}_AtteMask'.format(i+1), 'Part{}_visual'.format(part_num))

        map = simi_map[:,i,:].reshape(b,t,h*w)
        map = F.softmax(map*2, dim=-1)
        map = guiyihua(map).reshape(b, t, h, w)
        keshihua(map, raw_imgs, 'part_{}_SimiMap'.format(i+1), 'Part{}_visual'.format(part_num))

    # map = x_a.sum(1).reshape(b,t, h*w)
    # map = F.softmax(map, dim=-1)
    # map = guiyihua(map).reshape(b, t, h, w)
    # keshihua(map, raw_imgs, 'AppeFeat', 'batch{}'.format(batch))
    #
    # map = x_m.sum(1).reshape(b,t, h*w)
    # map = F.softmax(map*10, dim=-1)
    # map = guiyihua(map).reshape(b, t, h, w)
    # keshihua(map, raw_imgs, 'MotionFeat', 'batch{}'.format(batch))

    # mask = simi_map[0][:, 60, :].reshape(b, t, h*w)  ### b*thw*thw
    # mask = F.softmax(mask*0.01, dim=-1).reshape(b, t, h, w)
    # mask = guiyihua(mask).reshape(b, t, h, w)
    # keshihua(mask, raw_imgs, 'ST_Dependencies', 'batch{}'.format(batch))
    #
    # mask = simi_map[1][:, 60, :].reshape(b, t, h * w)  ### b*hw*thw
    # mask = F.softmax(mask*0.01, dim=-1).reshape(b, t, h, w)
    # mask = guiyihua(mask).reshape(b, t, h, w)
    # keshihua(mask, raw_imgs, 'S_Dependencies', 'batch{}'.format(batch))

    # mask = simi_map[2][:, 3, :].reshape(b, t, h * w)  ### b*t*thw
    # mask = F.softmax(mask*6, dim=-1).reshape(b, t, h, w)
    # mask = guiyihua(mask).reshape(b, t, h, w)
    # keshihua(mask, raw_imgs, 'T_Dependencies', 'batch{}'.format(batch))
    #
    # mask = simi_map[3].reshape(b, t, h * w)  ### b*1*thw
    # mask = F.softmax(mask*10, dim=-1).reshape(b, t, h, w)
    # mask = guiyihua(mask).reshape(b, t, h, w)
    # keshihua(mask, raw_imgs, 'G_Dependencies', 'batch{}'.format(batch))

    # map = motion_map[0].sum(1).reshape(b, t, h * w)
    # # map = F.softmax(map*1, dim=-1)
    # map = guiyihua(map).reshape(b, t, h, w)
    # keshihua(map, raw_imgs, 'ForwardMotion', 'batch{}'.format(batch))
    # #
    # map = motion_map[1].sum(1).reshape(b, t, h * w)
    # # map = F.softmax(map*1, dim=-1)
    # map = guiyihua(map).reshape(b, t, h, w)
    # keshihua(map, raw_imgs, 'BackwardMotion', 'batch{}'.format(batch))





