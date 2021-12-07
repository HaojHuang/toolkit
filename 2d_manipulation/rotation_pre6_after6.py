import numpy as np
import torch
import kornia as K
import matplotlib.pyplot as plt
import torchvision

def imshow( input: torch.Tensor, size: tuple = None):
    # A batch N*3*H*W: tensor type
    input_ = input[:, 0:3, :, :]

    out = torchvision.utils.make_grid(input_, nrow=6, padding=5)
    out_np: np.ndarray = K.utils.tensor_to_image(out)
    plt.figure(figsize=size)
    plt.imshow(out_np)
    plt.axis('off')
    plt.show()

img = plt.imread('./screenshot.png')
print(img.shape)
img = img[:96,:96,:3]
print(img.shape)
#plt.imshow(img)
#plt.show()

in_tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(dim=0).repeat(6,1,1,1)
print(in_tensor.size())

#6 images rotated [0,10,20,30,40,50] degress
pre_rotate_images = K.geometry.rotate(in_tensor, torch.from_numpy(np.linspace(0., 60., 6,endpoint=False, dtype=np.float32)),mode='bilinear',)
#print(np.linspace(0., 60., 6,endpoint=False, dtype=np.float32))
#print(pre_rotate_images.size())
#imshow(pre_rotate,size=(6,1))

# repeat 6 images in to 36
pre_rotate_images = pre_rotate_images.repeat(6,1,1,1)

#imshow(pre_rotate_2,size=(6,6))

after_rotate_degrees = torch.from_numpy(np.linspace(0., 360., 6, endpoint=False, dtype=np.float32)).unsqueeze(dim=-1)
after_rotate_degrees = after_rotate_degrees.repeat(1,6).view(-1)
print(after_rotate_degrees)
after_rotate = K.geometry.rotate(pre_rotate_images, after_rotate_degrees, mode='nearest',)

# conduct a crop to remove the side effects caused by the rotation
pivot = after_rotate.shape[-1]/2
half_length = 32
l, r = int(pivot - half_length), int(pivot + half_length + 1)
b, u = int(pivot - half_length), int(pivot + half_length + 1)
print(l,r)

after_rotate = after_rotate[:,:,l:r,b:u]
imshow(after_rotate,size=(6,6))
