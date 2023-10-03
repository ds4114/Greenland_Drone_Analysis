import numpy as np
import cv2
import torchvision.transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
print(torch.cuda.is_available())

from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d, MaxUnpool2d
from torch.nn import ReLU, Sigmoid, Softmax
from torch.nn import LogSoftmax
from torch import flatten
from torch.nn import MSELoss
import torch.optim as optim
from tqdm import tqdm

im1G = cv2.imread('point660map3_transparent_mosaic_green.tif', cv2.IMREAD_GRAYSCALE)
im1R = cv2.imread('point660map3_transparent_mosaic_red.tif', cv2.IMREAD_GRAYSCALE)
im1NIR = cv2.imread('point660map3_transparent_mosaic_nir.tif', cv2.IMREAD_GRAYSCALE)
im1RE = cv2.imread('point660map3_transparent_mosaic_red edge.tif', cv2.IMREAD_GRAYSCALE)

full_image = np.array([im1G,im1R,im1NIR,im1RE])

class FullImageDataset(Dataset):
    """Note, we will be chopping off part of the image"""

    def __init__(self, full_image, num_pixels):
      full_image = full_image.T
      full_image = cv2.pyrDown(full_image)
      full_image = cv2.pyrDown(full_image)
      full_image = full_image.T

      rows_resized = full_image.shape[1] % num_pixels
      columns_resized = full_image.shape[2] % num_pixels
      self.image = full_image[:,:-rows_resized, :-columns_resized]

      self.size = (self.image.shape[2] // num_pixels, self.image.shape[1] // num_pixels)
      self.num_pixels = num_pixels
      print(self.image.shape, self.size)
      self.m_val = np.max(self.image)


    def __len__(self):
        return self.size[0] * self.size[1]

    def __getitem__(self, idx):
      row = idx // self.size[0]
      col = idx % self.size[1]
      start_row = row * self.num_pixels
      start_col = col * self.num_pixels
      partial = self.image[:,start_row:start_row + self.num_pixels, start_col:start_col + self.num_pixels]
      tensor_p = torch.tensor(partial, dtype=torch.float)
      tensor_p = tensor_p / 255
      return tensor_p

ds = FullImageDataset(full_image, 500)
dl = DataLoader(ds, 16)

class FeatureNet(Module):

  def __init__(self):
    super(FeatureNet, self).__init__()
    self.conv1 = Conv2d(4, 8, 21, padding=10)
    self.max_pool = MaxPool2d(kernel_size=(2,2), stride=(2,2), return_indices=True)
    self.conv2 = Conv2d(8, 16, 21, padding=10)
    self.up_conv1 = Conv2d(16, 16, 21, padding=10)
    self.max_unpool = MaxUnpool2d(kernel_size=(2,2), stride=(2,2))
    self.up_conv2 = Conv2d(24, 8, 21, padding=10)
    self.up_conv3 = Conv2d(12, 4, 21, padding=10)

    self.to_sm = Conv2d(4, 5, 11, padding=5)

    self.conv3 = Conv2d(5, 8, 21, padding=10)
    self.conv4 = Conv2d(8, 16, 21, padding=10)
    self.up_conv4 = Conv2d(16, 16, 21, padding=10)
    self.max_unpool = MaxUnpool2d(kernel_size=(2,2), stride=(2,2))
    self.up_conv5 = Conv2d(24, 8, 21, padding=10)
    self.up_conv6 = Conv2d(13, 4, 21, padding=10)

    self.to_full = Conv2d(4, 4, 21, padding=10)

    self.relu = ReLU()
    self.sig = Sigmoid()
    self.sm = Softmax(dim=1)

  def forward(self, x):
    # #full = self.sig(self.convt(x))
    # x = self.relu(self.conv1(x))
    # # x = self.relu(self.conv2(x))
    # # x = self.relu(self.conv3(x))
    # # x = self.relu(self.conv4(x))
    # sing = self.sm(self.conv5(x))
    # full = self.sig(self.conv6(sing))


    orig, ind1 = self.max_pool(self.relu(self.conv1(x)))
    c1, ind2 = self.max_pool(self.relu(self.conv2(orig)))
    c2 = self.relu(self.up_conv1(c1))
    c3 = self.max_unpool(c2, indices=ind2)
    c3 = torch.cat([c3, orig], dim=1)
    c4 = self.max_unpool(self.relu(self.up_conv2(c3)), indices=ind1)
    c4 = torch.cat([c4, x], dim=1)
    c5 = self.relu(self.up_conv3(c4))
    preds = self.sm(self.to_sm(c5))

    d0, ind3 = self.max_pool(self.relu(self.conv3(preds)))
    d1, ind4 = self.max_pool(self.relu(self.conv4(d0)))
    d2 = self.relu(self.up_conv4(d1))
    d3 = self.max_unpool(d2, indices=ind4)
    d3 = torch.cat([d3, d0], dim=1)
    d4 = self.max_unpool(self.relu(self.up_conv5(d3)), indices=ind3)
    d4 = torch.cat([d4, preds], dim=1)
    d5 = self.relu(self.up_conv6(d4))

    full = self.sig(self.to_full(d5))


    return full, preds

criterion = MSELoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
learning_rate = 0.0001
model = FeatureNet()
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 200

blur = torchvision.transforms.GaussianBlur(51)

for epoch in tqdm(range(num_epochs)):
    running_loss = 0.0
    blur_loss = 0
    i = 0
    for data in dl:
        #print(i)
        i += 1
        inputs = data
        inputs = inputs.to(device)

        optimizer.zero_grad()

        outputs,classes = model(inputs)

        blur_classes = blur(classes)

        smoothness_loss = criterion(classes, blur_classes)*.005
        smoothness_loss += torch.mean(.25 - torch.pow(classes - .5,2))*.005
        #smoothness_loss += torch.mean(ext_vals)

        blur_loss += smoothness_loss.item()


        acc_loss = criterion(outputs, inputs)
        loss = acc_loss + smoothness_loss
        loss.backward()

        optimizer.step()
        running_loss += acc_loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {acc_loss / len(dl)} Blur Loss: {blur_loss / len(dl)}')

print('Finished Training')

example_image = ds.__getitem__(7)


cv2.imshow("Example",example_image[0,:,:].numpy())

ex = example_image.to(device)
ex = torch.reshape(ex, (1,ex.shape[0], ex.shape[1], ex.shape[2]))
pred, sing = model(ex)

pred, sing = pred.detach().cpu(), sing.detach().cpu()#sing.detach().cpu()
pred = pred.squeeze()
cv2.imshow("Prediction",pred[0,:,:].numpy())
cv2.imshow("Class 1",sing.squeeze()[0,:,:].numpy())
cv2.imshow("Class 2",sing.squeeze()[1,:,:].numpy())
cv2.imshow("Class 3",sing.squeeze()[2,:,:].numpy())
cv2.imshow("Class 4",sing.squeeze()[3,:,:].numpy())
cv2.imshow("Class 5",sing.squeeze()[4,:,:].numpy())
# print(sing)
# print(pred)
# print(example_image)
cv2.waitKey(50000)
cv2.destroyAllWindows()