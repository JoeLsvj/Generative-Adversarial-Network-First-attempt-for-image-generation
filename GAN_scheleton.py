import torch
from torch import nn
import torch.optim as optim

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

		## a seconda che si usi la BCEWithLogitsLoss o la BCE bisogna aggiungere uno strato di
		# normalizzazione tra 0 e 1 finale (e.g. Sigmoid)
		self.predictor = nn.Sequential(
			nn.Linear(512 * 4**2, 4096),
			nn.BatchNorm1d(4096),
			nn.ReLU(),
			nn.Linear(4096, 1)
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

				img_size (int): la dimensione dell'immagine in ingresso (supposta quadrata; a volontà
					si può decidere di usare dimensione reali, bisogna modificare un po' di codice)

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

		## in questa rete ho diviso la creazione del vettore latente (parte precedente) e la ricostruzione a partire
		# dal vettore latente; in questo modo, se volete visualizzarlo o adoperare loss diverse per l'encoder,
		# potreste farlo senza grosse rivoluzioni al codice
		self.dec_bottleneck = nn.Sequential(
			nn.Linear(nz, 4096),
			nn.BatchNorm1d(4096),
			nn.ReLU(),
			nn.Linear(4096, 512 * 4**2)
		)

		# nota: se usate immagini non multiple di 16, potreste avere problemi
		self.dec_pool = nn.AdaptiveAvgPool2d((img_size //16, img_size //16))

		## la sequenza di convoluzioni trasposte che ricostruisce l'immagine; nel caso più semplice di GAN,
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
		# l'output del modulo successivo è il vettore latente (tipicamente z, se volete portarlo fuori)
		z = self.enc_bottleneck(x)
		x = self.dec_bottleneck(z)
		x = x.view(-1, 512, 4, 4)

		x = self.dec_pool(x)
		x = self.decoder(x)

		return x


# Root directory for dataset
dataroot = "data/celeba"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input) (generalmente 100)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")

netD = Discriminator()

netG = Generator(nz=nz, img_size=image_size)
# (meglio scegliere img_size multiplo di 16, per come è scritto il codice del generatore)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
# the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# training
print("Inizio training...")
for t in range(epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (t, epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((t == epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
