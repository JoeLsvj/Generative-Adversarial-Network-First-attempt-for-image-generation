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

		self.adapool = nn.AdaptiveAvgPool2d((4,4))

		self.predictor = nn.Sequential(
			nn.Linear(512 * 4**2, 4096),
			nn.BatchNorm1d(4096),
			nn.ReLU(),
			nn.Linear(4096, 1)
		)


	def forward(self, x):

		x = self.features(x)
		x = self.adapool(x)

		x = x.view(-1, 512 * 4**2)
		x = self.predictor(x)

		return x

class Generator(nn.Module):

	def __init__(self, nz, img_size):
		'''
			Args:

				nz (int): la dimensione del vettore latente

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

		self.enc_bottleneck = nn.Sequential(
			nn.Linear(512 * 4**2, 4096),
			nn.BatchNorm1d(4096),
			nn.ReLU(),
			nn.Linear(4096, nz)
		)

		self.dec_bottleneck = nn.Sequential(
			nn.Linear(nz, 4096),
			nn.BatchNorm1d(4096),
			nn.ReLU(),
			nn.Linear(4096, 512 * 4**2)
		)

		self.dec_pool = nn.AdaptiveAvgPool2d((img_size //16, img_size //16))

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
		z = self.enc_bottleneck(x)
		x = self.dec_bottleneck(z)
		x = x.view(-1, 512, 4, 4)

		x = self.dec_pool(x)
		x = self.decoder(x)

		return x
