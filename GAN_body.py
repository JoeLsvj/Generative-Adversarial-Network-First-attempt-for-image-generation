import torch
import os
from torch import nn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

def img_tool(image, W=0, H=0, show=False, to_tensor=False):
    # Funzione per importare, ridimensionare e mostrare un immagine.
    # 'image': ndarray numpy o stringa del path (assoluto o relativo) del file immagine in ingresso
    # 'H', 'W': dimensioni desiderate per il resize (lo spazio residuo viene riempito di nero),
    # se non sono specificati non viene fatto il resize 
    # 'rot_angle': float o int, angolo di rotazione, default=0
    # 'noise_percent': percentuale di rumore gaussiano aggiunto
    # 'show': bool, se è True mostra l'immagine con opencv
    # 'to_tensor': bool, se è True restituisce un torch.Tensor (1,1,H,W) anzichè un np.ndarray
    if type(image)==str:
        img = cv.imread(image, 0)
        if img is None:
            # provo a ricostruire il path completo
            # (utile se la cwd è diversa dalla directory in cui si trova il .py in esecuzione)
            target_path = os.path.dirname(__file__)
            image = os.path.join(target_path, image)
            img = cv.imread(image, 0)
        if img is None:
            raise NameError("wrong image name or path")
    elif type(image)==np.ndarray:
        img = image
    else:
        raise TypeError("image argument must be of type str or np.ndarray")
    if H==0 and W==0:
        H, W = img.shape[:2]
    # resize
    img_shape = img.shape
    rapporto = np.minimum(H/img_shape[0], W/img_shape[1])
    new_img_shape = (int(img_shape[0]*rapporto), int(img_shape[1]*rapporto))
    img = cv.resize(img, (new_img_shape[1], new_img_shape[0]))
    blank = np.zeros((H, W), dtype=np.uint8) # type per leggere la matrice immagine a 255 livelli
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            blank[i,j] = img[i,j]
    img = blank
    # show
    if show:
        cv.imshow("Resized image", img)
        cv.waitKey(0)
        cv.destroyAllWindows() 
    # to tensor
    if to_tensor:
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=0)
        img = torch.Tensor(img)
    return img

class TensorBoard_handle(object):
    # To enable the debugger in TensorBoard, use the flag: --debugger_port <port_number>
    # tensorboard --logdir=runs to run tensorboard from terminal
    def __init__ (self, DataLoader, dir_name):
        path = os.path.join(dir_name, "tensorboard")
        self.writer = SummaryWriter(path)
        self.dataloader = DataLoader

    @staticmethod
    def tensorboard_imshow(img, one_channel=False):
        img = img.cpu()
        if one_channel:
            img = img.mean(dim=0)
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        if one_channel:
            plt.imshow(npimg, cmap="Greys")
        else:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def make_images_grid (self):
        dataiter = iter(self.dataloader)
        images, _ = dataiter.next()
        images = images[:8] 
        img_grid = torchvision.utils.make_grid(images)
        return images, img_grid
        # tensorboard.writer.add_image('some_fashion_mnist_images', img_grid)
        # tensorboard.writer.add_graph(model, images)
        # tensorboard.writer.close()

    def add_scalar(self, tag, scalar, step):
        # scalar(float) è il float da caricare ad ogni step(int)
        self.writer.add_scalar(tag, scalar, step)

    def add_n_embedding(self, dataset, n=100):
        # Selects n random datapoints and their corresponding labels from a dataset
        inputz, targetz = dataset[:][0], dataset[:][1]
        assert len(inputz) == len(targetz)
        perm = torch.randperm(len(inputz))
        images, labels = inputz[perm][:n], targetz[perm][:n]
        assert len(images) == len(labels)
        # select random images and their target indices and get the class labels for each image
        class_labels = [self.classes[i] for i in labels]
        features = images.view(-1, 28 * 28)
        self.writer.add_embedding(mat=features, metadata=class_labels, label_img=images.unsqueeze(1))
        self.writer.close()

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # per ridurre la dimensione finale dell'immagine a 512 x 4 x 4
        self.adapool = nn.AdaptiveAvgPool2d((4,4))

        # a seconda che si usi la BCEWithLogitsLoss o la BCE bisogna aggiungere uno strato di
        # normalizzazione tra 0 e 1 finale (e.g. Sigmoid)
        self.predictor = nn.Sequential(
            nn.Linear(512 * 4**2, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adapool(x)
        # trasforma il tensore in un vettore, preservando la divisione per batch
        x = x.view(-1, 512 * 4**2)
        x = self.predictor(x)
        return x

class Generator(nn.Module):

    def __init__(self, nz, img_size):
        '''
            Args:

                nz (int): la dimensione del vettore latente; tipicamente 100,
                    maggiore è la dimensione, maggiore il numero di dati conservati, maggiore
                    il tempo di apprendimento ed inferenza;
                    non sempre a spazio latente di dimensione maggiore corrisponde ricostruzione migliore

                img_size (int): la dimensione dell'immagine in ingresso

        '''
        super(Generator, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # img_size = img_size

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # img_size = img_size /2

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # img_size = img_size /4

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # img_size = img_size /8

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # img_size = img_size /16
        )

        self.enc_pool = nn.AdaptiveAvgPool2d((4,4))

        # gli strati lineari che calcolano il vettore latente
        self.enc_bottleneck = nn.Sequential(
            nn.Linear(512 * 4**2, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, nz)
        )

        # in questa rete ho diviso la creazione del vettore latente (parte precedente) e la ricostruzione a partire dal vettore latente
        self.dec_bottleneck = nn.Sequential(
            nn.Linear(nz, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 512 * 4**2)
        )

        # nota: possibili problemi per immagini non multiple di 16
        self.dec_pool = nn.AdaptiveAvgPool2d((img_size //16, img_size //16))

        # la sequenza di convoluzioni trasposte che ricostruisce l'immagine; nel caso più semplice di GAN,
        # questa componente e il modulo dec_bottleneck precedente rappresentano l'intero generatore
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # img_size = img_size *8

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # img_size = img_size *4

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # img_size = img_size *2

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # img_size = img_size

            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Tanh(),
            # img_size = img_size
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.enc_pool(x)
        x = x.view(-1, 512 * 4**2)
        # l'output del modulo successivo è il vettore latente (z)
        z = self.enc_bottleneck(x)
        x = self.dec_bottleneck(z)
        x = x.view(-1, 512, 4, 4)
        x = self.dec_pool(x)
        x = self.decoder(x)
        return x

def train_Discriminator(real_data, fake_data, label, netD, netG, loss, optimizer):

    # Train with all-real batch
    netD.zero_grad()
    # Forward pass real batch through D
    output = netD(real_data).view(-1)
    # Calculate loss on all-real batch
    errD_real = loss(output, label)
    # Calculate gradients for D in backward pass
    errD_real.backward()
    D_x = output.mean().item()

    # Train with all-fake batch
    label.fill_(fake_label)
    # Classify all fake batch with D
    output = netD(fake_data.detach()).view(-1)
    # Calculate D's loss on the all-fake batch
    errD_fake = loss(output, label)
    # Calculate the gradients for this batch
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    # Add the gradients from the all-real and all-fake batches
    errD = errD_real + errD_fake
    if errD > 0.5:
        # Update D
        optimizer.step()
    return errD, D_G_z1, D_x

def train_Generator(real_data, fake_data, label, netD, netG, loss1, loss2, optimizer):
    netG.zero_grad()
    label.fill_(real_label) # fake labels are real for generator cost
    # Since we just updated D, perform another forward pass of all-fake batch through D
    output = netD(fake_data).view(-1)
    # Calculate G's loss based on this output
    errG_1 = loss1(output, label)
    errG_2 = loss2(fake_data, real_data)
    errG = errG_2
    # Calculate gradients for G
    errG.backward()
    D_G_z2 = output.mean().item()
    # Update G
    optimizer.step()
    return errG, D_G_z2

def main():
    # Root directory for dataset
    global rootdir, dataroot, device, real_label, fake_label
    rootdir = os.path.dirname(__file__)
    dataroot = os.path.join(rootdir, "bottle_dataset/train")

    ## parametri:
    workers = 2    # Number of workers for dataloader
    batch_size = 8    # Batch size during training
    image_size = 512    # Spatial size of training images. All images will be resized to this size using a transformer
    nc = 3     # Number of channels in the training images. For color images this is 3
    nz = 100    # Size of z latent vector (i.e. size of generator input) (generalmente 100)
    ngf = 64    # Size of feature maps in generator
    ndf = 64    # Size of feature maps in discriminator
    epochs = 50    # Number of training epochs
    lr = 0.0002    # Learning rate for optimizers
    beta1 = 0.5    # Beta1 hyperparam for Adam optimizers
    ngpu = 1    # Number of GPUs available. Use 0 for CPU mode

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")

    # Create the dataset
    dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)

    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

    netD = Discriminator().to(device)
    netG = Generator(nz=nz, img_size=image_size).to(device)
    # (meglio scegliere img_size multiplo di 16, per come è scritto il codice del generatore)

    # Initialize BCELoss function
    criterion = nn.BCELoss()
    L1 = nn.L1Loss()

    # Create batch of latent vectors that we will use to visualize
    # the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    tensorboard = TensorBoard_handle(dataloader, rootdir)
    images, img_grid = tensorboard.make_images_grid()
    tensorboard.writer.add_image('some_images', img_grid)
    # tensorboard.writer.add_graph(netD, images.to(device))
    # tensorboard.writer.add_graph(netG, images.to(device))

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    # training
    print("Inizio training...")
    for t in range(epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            # format batch
            real_data = data[0].to(device)
            # Generate fake images with generator 
            fake_data = netG(real_data)
            b_size = real_data.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            img1 = real_data
            vutils.save_image(img1, rootdir + "real.png")

            fake_grid = torchvision.utils.make_grid(fake_data)
            tensorboard.writer.add_image('fake', fake_grid)
            errD, D_G_z1, D_x = train_Discriminator(real_data=real_data, fake_data=fake_data, label=label, loss=criterion, netD=netD, netG=netG, optimizer=optimizerD)

            # (2) Update G network: maximize log(D(G(z)))
            errG, D_G_z2 = train_Generator(real_data=real_data, fake_data=fake_data, label=label, netD=netD, netG=netG, loss1=criterion, loss2=L1, optimizer=optimizerG)

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (t, epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                
                tensorboard.add_scalar('errD', errD.item(), i+t*len(dataloader))
                tensorboard.add_scalar('errG', errG.item(), i+t*len(dataloader))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 100 == 0) or ((t == epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake_data = netG(real_data).detach().cpu()
                img_list.append(vutils.make_grid(fake_data, padding=2, normalize=True))
                img1 = fake_data[0]
                vutils.save_image(img1, rootdir + "img1.png")
            iters += 1

main()
