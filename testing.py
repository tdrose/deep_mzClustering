from mz_clustering.CAE import conv2d_hout, conv2d_hwout, conv2d_wout, CAE
from mz_clustering.cnnClust import cnnClust
from mz_clustering.clustering import Clustering
import numpy as np
import torch
import torch.nn.functional as functional
from metaspace import SMInstance

test1 = False
test2 = False
test3 = False
test4 = False
test5 = True

# width=4, height=5
width = 35
height = 60
n_img = 100
num_clust = 7

imgs = np.random.randn(width*height*n_img).reshape((-1, height, width))
imgt = torch.tensor(imgs, dtype=torch.float)
imgt2 = imgt.reshape((-1, 1, height, width))


cae = CAE(height=height, width=width)
clust = cnnClust(num_clust=num_clust, height=height, width=width)

if test1:

    l1 = cae.conv1(imgt2)
    l1h = conv2d_hout(height=height, padding=(0, 0), dilation=(1, 1), kernel_size=(3, 3), stride=(2, 2))
    l1w = conv2d_wout(width=width, padding=(0, 0), dilation=(1, 1), kernel_size=(3, 3), stride=(2, 2))

    l2 = cae.conv2(l1)
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

    # l3 = model.conv_trans1(l2)
    # l4 = model.conv_trans2(l3)
    l33 = cae.convtrans1(l2, output_size=(l1h, l1w))
    l44 = cae.convtrans2(l33, output_size=(height, width))

    # print(f"trans 1 shape: {l3.shape}")
    print(f"trans 1 shape: {l33.shape}")
    print(f"trans 2 shape: {l44.shape}")

    z = cae.encode(imgt2)
    out = cae.decode(z)
    print(f"encoded: {z.shape}")
    print(f"Encoded: {out.shape}")

if test2:

    l1 = clust.conv1(imgt2)
    print(f"input shape: {imgt2.shape}")
    l2 = clust.conv2(l1)
    l3 = clust.conv3(l2)
    l4 = clust.conv4(l3)
    l5 = clust.conv5(l4)
    l6 = clust.conv6(l5)

    l2h, l2w = conv2d_hwout(height=height, width=width, padding=(0, 0),
                            dilation=(1, 1), kernel_size=(2, 2), stride=(1, 1))
    l2hh, l2ww = conv2d_hwout(height=l2h, width=l2w, padding=(0, 0),
                              dilation=(1, 1), kernel_size=(2, 2), stride=(2, 2))

    l3h, l3w = conv2d_hwout(height=l2hh, width=l2ww, padding=(0, 0),
                            dilation=(1, 1), kernel_size=(2, 2), stride=(1, 1))
    l4h, l4w = conv2d_hwout(height=l3h, width=l3w, padding=(0, 0),
                            dilation=(1, 1), kernel_size=(3, 3), stride=(1, 1))

    l5h, l5w = conv2d_hwout(height=l4h, width=l4w, padding=(0, 0),
                            dilation=(1, 1), kernel_size=(3, 3), stride=(1, 1))
    l5hh, l5ww = conv2d_hwout(height=l5h, width=l5w, padding=(0, 0),
                              dilation=(1, 1), kernel_size=(2, 2), stride=(2, 2))

    l6h, l6w = conv2d_hwout(height=l5hh, width=l5ww, padding=(0, 0),
                            dilation=(1, 1), kernel_size=(3, 3), stride=(1, 1))

    print(f"L2 shape: {l6.shape}")
    print(f"Calculated height: {l6h}")
    print(f"Calculated width: {l6w}")

if test3:
    out = clust.forward(imgt2)
    print(out.shape)

if test4:
    clustering = Clustering(imgs,
                            num_cluster=7,
                            height=height, width=width,
                            lr=0.0001,
                            batch_size=10,
                            knn=True, k=10,
                            use_gpu=False)
    cae = CAE(train_mode=True, height=height, width=width)
    clust = cnnClust(num_clust=num_clust, height=height, width=width)

    batch, labels, index = clustering.get_batch(clustering.image_data, clustering.batch_size, )
    print(index)
    train_x = torch.Tensor(batch)
    train_x = train_x.reshape((-1, 1, height, width))
    print(train_x.shape)

    x = cae.forward(train_x)
    features = clust.forward(x)

    features = functional.normalize(features, p=2, dim=-1)
    # Another normalization !?
    features = features / features.norm(dim=1)[:, None]
    # Similarity as defined in formula 2 of the paper
    sim_mat = torch.matmul(features, torch.transpose(features, 0, 1))
    sim_numpy = sim_mat.cpu().detach().numpy()
    # Get all sim values from the batch excluding the diagonal
    # Todo: why not complete batch size?
    tmp2 = [sim_numpy[i][j] for i in range(0, clustering.batch_size - 1)
            for j in range(clustering.batch_size - 1) if i != j]
    # Compute upper and lower percentiles according to uu & ll
    uu = 98
    ll = 46
    ub = np.percentile(tmp2, uu)
    lb = np.percentile(tmp2, ll)

    print(sum(clustering.knn_adj))
    pos_loc = (sim_mat >= ub).type("torch.FloatTensor")
    pos_loc.cpu().detach().numpy()
    print(pos_loc.cpu().detach().numpy())

if test5:
    # Example metaspace dataset
    print("Downloading data")
    sm = SMInstance()
    dsid = "2019-08-02_22h43m20s"
    ds = sm.dataset(id=dsid)
    tmp = ds.all_annotation_images(fdr=0.2, database=("HMDB", "v4"))
    tmp = np.array([x._images[0] for x in tmp])
    mheight = tmp.shape[1]
    mwidth = tmp.shape[2]
    print("Starting training")
    clustering = Clustering(tmp,
                            num_cluster=7,
                            height=mheight, width=mwidth,
                            lr=0.0001,
                            batch_size=100,
                            knn=True, k=10,
                            use_gpu=False)

    cae, clust = clustering.train()
