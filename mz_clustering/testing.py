from mz_clustering.CAE import conv2d_hout, conv2d_wout, CAE
import numpy as np
import torch


# width=4, height=5
width = 40
height = 50
n_img = 11

imgs = np.arange(width*height*n_img).reshape((-1, height, width))

imgt = torch.tensor(imgs, dtype=torch.float)

imgt2 = imgt.reshape((-1, 1, height, width))


model = CAE(height=height, width=width)

l1 = model.conv1(imgt2)
l1h = conv2d_hout(height=height, padding=(0, 0), dilation=(1, 1), kernel_size=(3, 3), stride=(2, 2))
l1w = conv2d_wout(width=width, padding=(0, 0), dilation=(1, 1), kernel_size=(3, 3), stride=(2, 2))

l2 = model.conv2(l1)
l2h = conv2d_hout(height=l1h, padding=(0, 0), dilation=(1, 1), kernel_size=(3, 3), stride=(3, 3))
l2w = conv2d_wout(width=l1w, padding=(0, 0), dilation=(1, 1), kernel_size=(3, 3), stride=(3, 3))

print(f"input shape: {imgt2.shape}")
print()
print(f"L1 shape: {l1.shape}")
print(f"Calculated height: {l1h}")
print(f"Calculated width: {l1w}")
print()
print(f"L2 shape: {l2.shape}")
print(f"Calculated height: {l2h}")
print(f"Calculated width: {l2w}")
print()
print(f"Vectorised shape: {l2.view(l2.size(0), -1).shape}")
print(f"Calculated shape: {l2h*l2w*16}")

#l3 = model.conv_trans1(l2)
#l4 = model.conv_trans2(l3)
l33 = model.convtrans1(l2, output_size=(l1h, l1w))
l44 = model.convtrans2(l33, output_size=(height, width))

#print(f"trans 1 shape: {l3.shape}")
print(f"trans 1 shape: {l33.shape}")
print(f"trans 2 shape: {l44.shape}")

z = model.encode(imgt2)
out = model.decode(z)
print(f"encoded: {z.shape}")
print(f"Encoded: {out.shape}")



