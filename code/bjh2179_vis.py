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
from torch.nn import MSELoss,CrossEntropyLoss
import torch.optim as optim
from tqdm import tqdm
from scipy import ndimage
import os

import torchvision.transforms as transforms


def make_image(height="top"):
    pref = "point660map2"
    if height == "med":
        pref = "point660map3"
    elif height == "bot":
        pref = "map3a"

    suffix = ["green", "red", "nir", "red edge"]
    ims = []
    for s in suffix:
        ims.append(cv2.imread(pref + "_transparent_mosaic_" + s + ".tif", cv2.IMREAD_GRAYSCALE).astype(np.float32))
    # print(min((ims[2] + ims[1]) + .01))
    #ndvi = np.rint(np.divide((ims[2] - ims[1]), (ims[2] + ims[1]) + .001))

    #ndvi = (ndvi - np.min(ndvi)).astype(np.float32)
    #ndvi /= np.max(ndvi)


    #ims.append(ndvi)

    full_image = np.array(ims)
    return full_image


class FullImageDataset(Dataset):
    """Note, we will be chopping off part of the image"""

    def __init__(self, full_image=None, num_pixels=None, train=True, dir=None, transform=None):
        self.transform = transform
        self.dir = dir
        if dir is None:
            rows_resized = full_image.shape[1] % num_pixels
            columns_resized = full_image.shape[2] % num_pixels
            image = full_image[:, :-rows_resized, :-columns_resized]

            if train:
                square = min(image.shape[1], image.shape[2])
                image = full_image[:, :square, :square]


            self.size = (image.shape[2] // num_pixels, image.shape[1] // num_pixels)
            self.num_pixels = num_pixels
            print(image.shape, self.size)
            self.m_val = np.max(image)

            for i in range(len(image)):
                image[i, :, :] = (image[i, :, :] - np.min(image[i, :, :])) / (np.max(
                    (image[i, :, :] - np.min(image[i, :, :]))) + .0000001)

            ims = []
            noise_ims = []
            for idx in range(self.size[0] * self.size[1]):
                row = idx // self.size[0]
                col = idx % self.size[1]
                start_row = row * self.num_pixels
                start_col = col * self.num_pixels
                partial = image[:, start_row:start_row + self.num_pixels, start_col:start_col + self.num_pixels]

                tensor_p = torch.tensor(partial, dtype=torch.float)
                ims.append(tensor_p)
                if train:
                    partial = partial.T
                    ims.append(torch.tensor(self.rotations(partial, 90).T, dtype=torch.float))
                    ims.append(torch.tensor(self.rotations(partial, 180).T, dtype=torch.float))
                    ims.append(torch.tensor(self.rotations(partial, 270).T, dtype=torch.float))
            if train:
                std = torch.ones(size=ims[0].shape) * .2
                for i in range(len(ims)):
                    noise = torch.normal(mean=0, std=std,dtype=torch.float).type_as(torch.float)
                    temp_image = ims[i] + noise
                    for j in range(len(temp_image)):
                        temp_image[j,:,:] = (temp_image[j,:,:] - torch.min(temp_image[j,:,:])) / (torch.max(temp_image[j,:,:] - torch.min(temp_image[j,:,:])) + .0000001)
                    noise_ims.append(temp_image)

                self.all_ims = ims + noise_ims
            else:
                self.all_ims = ims

            # for i in range(len(self.all_ims)):
            #     self.all_ims[i] = self.all_ims[i] - torch.min(self.all_ims[i]) / torch.max(self.all_ims[i] - torch.min(self.all_ims[i]))
        else:
            self.length = len([f for f in os.listdir(self.dir) if os.path.isfile(os.path.join(self.dir, f))])

    def rotations(self, im, degrees):
        return ndimage.rotate(im, degrees, axes=(1, 0))

    def __len__(self):
        if self.dir is None:
            return len(self.all_ims)
        return self.length

    def __getitem__(self, idx):

        # row = idx // self.size[1]
        # col = idx % self.size[0]
        # start_row = row * self.num_pixels
        # start_col = col * self.num_pixels
        # partial = self.image[:,start_row:start_row + self.num_pixels, start_col:start_col + self.num_pixels]
        # tensor_p = torch.tensor(partial, dtype=torch.float)

        # for i in range(len(tensor_p)):
        #     tensor_p[i,:,:] = (tensor_p[i,:,:] - torch.min(tensor_p[i,:,:])) / torch.max((tensor_p[i,:,:] - torch.min(tensor_p[i,:,:])))
        # tensor_p = tensor_p / 255
        if self.dir is None:
            if self.transform:
                return self.transform(self.all_ims[idx])
            return self.all_ims[idx]
        else:
            if self.transform:
                return self.transform(torch.from_numpy(cv2.imread(f"{self.dir}/image{idx}.tif", cv2.IMREAD_UNCHANGED).T/255).float())
            return torch.from_numpy(cv2.imread(f"{self.dir}/image{idx}.tif", cv2.IMREAD_UNCHANGED).T/255).float()


custom_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomErasing(p=.2)
])

ds = FullImageDataset(dir="train")

# for i in range(ds.__len__()):
#     im = ds.__getitem__(i)
#
#     cv2.imwrite(f"train/image{i}.tif", np.rint(im.squeeze().numpy() * 255).astype(np.uint8).T)

# np.savetxt("train_low.csv", ds.image)
dl = DataLoader(ds, 5, shuffle=True)
print("Training set created")


ds_val = FullImageDataset(dir="val", transform=custom_transforms)
dl_val = DataLoader(ds_val, 5, shuffle=True)

# for i in range(ds_val.__len__()):
#     im = ds_val.__getitem__(i)
#
#     cv2.imwrite(f"val/image{i}.tif", np.rint(im.squeeze().numpy() * 255).astype(np.uint8).T)
# np.savetxt("val_med.csv", ds_val.image)
print("Val set created")

# ds_test = FullImageDataset(make_image("top"), 500, train=False)
ds_test = FullImageDataset(dir="test")
# np.savetxt("test_top.csv", ds_test.image)

# for i in range(ds_test.__len__()):
#     im = ds_test.__getitem__(i)
#
#     cv2.imwrite(f"test/image{i}.tif", np.rint(im.squeeze().numpy() * 255).astype(np.uint8).T)

dl_test = DataLoader(ds_test, 4)
print("Test set created")


class FeatureNet(Module):

    def __init__(self):
        super(FeatureNet, self).__init__()
        self.conv1 = Conv2d(4, 16, 21, padding=10)
        self.max_pool = MaxPool2d(kernel_size=(2, 2), stride=(2, 2), return_indices=True)
        self.test = Conv2d(16, 16, 21, padding=10)
        self.conv2 = Conv2d(16, 16, 21, padding=10)
        self.up_conv1 = Conv2d(16, 16, 21, padding=10)
        self.max_unpool = MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2))
        self.up_conv2 = Conv2d(32, 16, 21, padding=10)

        self.up_conv3 = Conv2d(32, 16, 21, padding=10)
        self.test2 = Conv2d(16, 16, 21, padding=10)

        self.before_sm = Conv2d(16, 8, 21, padding=10)

        self.to_sm = Conv2d(8, 5, 11, padding=5)
        self.after = Conv2d(5, 16, 21, padding=10)

        self.conv3 = Conv2d(16, 16, 21, padding=10)
        self.conv4 = Conv2d(16, 16, 21, padding=10)
        self.up_conv4 = Conv2d(16, 16, 21, padding=10)
        self.max_unpool = MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2))
        self.up_conv5 = Conv2d(32, 16, 21, padding=10)
        self.up_conv6 = Conv2d(32, 16, 21, padding=10)

        self.before_final = Conv2d(16, 16, 21, padding=10)

        self.to_full = Conv2d(16, 4, 21, padding=10)

        self.relu = ReLU()
        self.sig = Sigmoid()
        self.sm = Softmax(dim=1)
        self.blur = torchvision.transforms.GaussianBlur(3, sigma=.1)

    def forward(self, x):
        # #full = self.sig(self.convt(x))
        # x = self.relu(self.conv1(x))
        # # x = self.relu(self.conv2(x))
        # # x = self.relu(self.conv3(x))
        # # x = self.relu(self.conv4(x))
        # sing = self.sm(self.conv5(x))
        # full = self.sig(self.conv6(sing))
        into = self.relu(self.test(self.relu(self.conv1(x))))
        orig, ind1 = self.max_pool(into)
        c1, ind2 = self.max_pool(self.relu(self.conv2(orig)))
        c2 = self.relu(self.up_conv1(c1))
        c3 = self.max_unpool(c2, indices=ind2)
        c3 = torch.cat([c3, orig], dim=1)
        c4 = self.max_unpool(self.relu(self.up_conv2(c3)), indices=ind1)

        c4 = torch.cat([c4, into], dim=1)

        c5 = self.relu(self.up_conv3(c4))
        upinto = self.relu(self.test2(c5))
        c6 = self.relu(self.before_sm(upinto))
        c7 = self.to_sm(c6)
        c7 *= (c7 >= 1/c7.shape[1])
        preds = self.sm(c7)

        aft = self.relu(self.after(preds))


        d0, ind3 = self.max_pool(self.relu(self.conv3(aft)))
        d1, ind4 = self.max_pool(self.relu(self.conv4(d0)))
        d2 = self.relu(self.up_conv4(d1))
        d3 = self.max_unpool(d2, indices=ind4)
        d3 = torch.cat([d3, d0], dim=1)
        d4 = self.max_unpool(self.relu(self.up_conv5(d3)), indices=ind3)
        d4 = torch.cat([d4, aft], dim=1)
        d5 = self.relu(self.up_conv6(d4))
        d6 = self.before_final(d5)
        d7 = self.relu(d6)

        full = self.sig(self.to_full(d7))

        return full, preds

torch.cuda.empty_cache()
device="cuda"
criterion = MSELoss()

def tv_regularization(predictions, alpha=.0005):
    return 0
    #predictions = torch.argmax(probabilites, dim=1)
    diff_x = predictions[:,:,1:,:] - predictions[:,:,:-1,:]
    diff_y = predictions[:,:,:,1:] - predictions[:,:,:,:-1]
    tv_loss = alpha * (torch.sum(torch.pow(diff_x,2)) + torch.sum(torch.pow(diff_y,2))) / predictions.shape[0] / predictions.shape[1] / predictions.shape[2] / predictions.shape[3]

    return tv_loss

def few_channel_penalty(predictions, alpha=.01):
    return 0
    means = torch.mean(predictions, dim=[2,3])
    batch_max_means = torch.max(means,dim=1).values - (1/predictions.shape[1])
    return alpha*torch.mean(batch_max_means)

def uncertianty_penalty(predictions, alpha=.0001):

    return torch.mean(.2500001 - torch.pow(predictions - .5, 2)) * alpha

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#
# model = FeatureNet()
# model.load_state_dict(torch.load("models/best_model.pickle"))
# model = model.to(device)
# model.eval()
# with torch.no_grad():
#     for i in range(22, 30):
#         example_image = ds_test.__getitem__(i)
#
#         cv2.imwrite(f"images_test/initial_images/item{i}green.tif", np.rint(example_image[0, :, :].numpy() * 255).astype(np.uint8))
#         cv2.imwrite(f"images_test/initial_images/item{i}red.tif", np.rint(example_image[1, :, :].numpy() * 255).astype(np.uint8))
#         cv2.imwrite(f"images_test/initial_images/item{i}nir.tif", np.rint(example_image[2, :, :].numpy() * 255).astype(np.uint8))
#         cv2.imwrite(f"images_test/initial_images/item{i}re.tif", np.rint(example_image[3, :, :].numpy() * 255).astype(np.uint8))
#         # cv2.imwrite(f"initial_images/item{i}ndvi.tif", np.rint(example_image[4, :, :].numpy()*255).astype(np.uint8))
#
#         ex = example_image.to(device)
#         ex = torch.reshape(ex, (1, ex.shape[0], ex.shape[1], ex.shape[2]))
#         pred, sing = model(ex)
#
#         pred, sing = pred.detach().cpu(), sing.detach().cpu()  # sing.detach().cpu()
#         pred = pred.squeeze()
#         cv2.imwrite(f"images_test/initial_images/item{i}PredG.tif", np.rint(pred[0, :, :].numpy() * 255).astype(np.uint8))
#         cv2.imwrite(f"images_test/initial_images/item{i}PredR.tif", np.rint(pred[1, :, :].numpy() * 255).astype(np.uint8))
#         cv2.imwrite(f"images_test/initial_images/item{i}PredNIR.tif", np.rint(pred[2, :, :].numpy() * 255).astype(np.uint8))
#         cv2.imwrite(f"images_test/initial_images/item{i}PredRE.tif", np.rint(pred[3, :, :].numpy() * 255).astype(np.uint8))
#
#         cv2.imwrite(f"images_test/initial_images/item{i}Class1.tif", np.rint(sing.squeeze()[0, :, :].numpy() * 255).astype(np.uint8))
#         cv2.imwrite(f"images_test/initial_images/item{i}Class2.tif", np.rint(sing.squeeze()[1, :, :].numpy() * 255).astype(np.uint8))
#         cv2.imwrite(f"images_test/initial_images/item{i}Class3.tif", np.rint(sing.squeeze()[2, :, :].numpy() * 255).astype(np.uint8))
#         cv2.imwrite(f"images_test/initial_images/item{i}Class4.tif", np.rint(sing.squeeze()[3, :, :].numpy() * 255).astype(np.uint8))
#         cv2.imwrite(f"images_test/initial_images/item{i}Class5.tif", np.rint(sing.squeeze()[4, :, :].numpy() * 255).astype(np.uint8))
#         # cv2.imwrite(f"initial_images/item{i}Class6.tif", np.rint(sing.squeeze()[5, :, :].numpy() * 255).astype(np.uint8))
#         # cv2.imwrite(f"initial_images/item{i}Class7.tif", np.rint(sing.squeeze()[6, :, :].numpy() * 255).astype(np.uint8))
#         # cv2.imwrite(f"initial_images/item{i}Class8.tif", np.rint(sing.squeeze()[7, :, :].numpy() * 255).astype(np.uint8))
#         # cv2.imwrite(f"initial_images/item{i}Class9.tif", np.rint(sing.squeeze()[8, :, :].numpy() * 255).astype(np.uint8))
#         # cv2.imwrite(f"initial_images/item{i}Class10.tif", np.rint(sing.squeeze()[9, :, :].numpy() * 255).astype(np.uint8))


def Soft_Normalized_Cut_Loss(image, classes, r=1, s2=1):
    # Long lines because I keep running out of memory
    K = classes.shape[0]

    def weight(u,s):
        ret = torch.exp(torch.sum(torch.pow((u-s.view(u.shape[0],u.shape[1],1,1).expand((u.shape[0],u.shape[1],u.shape[2],u.shape[3]))),2), dim=(1))/s2)
        ret[ret < r] = 0
        return ret

    t_loss = 0

    for k in range(K):
        for i in range(image.shape[2]):
            for j in range(image.shape[3]):
                w = weight(image, image[:, :, i, j])
                t_loss += torch.sum(w * classes[:,k,i,j].view(classes.shape[0],1,1).expand((classes.shape[0],classes.shape[2],classes.shape[3])) * classes[:,k,:,:])/torch.sum(w * classes[:,k,i,j].view(classes.shape[0],1,1).expand((classes.shape[0],classes.shape[2],classes.shape[3])))
                print(t_loss)

    return K - t_loss


def Modified_Soft_Normalized_Cut_Loss(image, classes):
    K = classes.shape[1]

    # def weighted_average(values, weights):
    #     updated = torch.where(weights < .000001, .000001, weights)
    #     val = torch.sum(updated*values)/torch.sum(updated)
    #     return val
    def avg(mask,values, weights):
        mask[mask == 0] = .00001
        values[values == 0] = .00001
        weights[weights == 0] = .00001


        mask = torch.stack([mask for i in range(values.shape[1])], dim=1)
        masked = values * weights
        return torch.mean(torch.sum(masked, dim=(0,2,3))) / (.000001 + torch.sum(weights))

    def weighted_average(mask, values, weights):


        weights[weights == 0] = .00001
        mask[mask == 0] = .00001
        values[values == 0] = .00001


        mask = torch.stack([mask for i in range(values.shape[1])], dim=1)
        masked = mask * values
        non_zero = torch.sum(mask[0])
        avg = torch.sum(masked, dim=(0,2,3)) / (.000001 + non_zero)
        assert values.shape[1] == avg.shape[0]

        for i in range(avg.shape[0]):
            masked[:,i,:,:] -= avg[i]

        masked *= weights * mask

        var = torch.pow(masked,2)

        scaled_vars = torch.sum(var) / (torch.sum(mask) + .000001)

        return scaled_vars

    def distance(values, classes, k):

        max_classes = torch.argmax(classes, dim=1, keepdim=True)
        adjusted_max = torch.where(max_classes != k, 0, max_classes)
        mask = torch.where(adjusted_max == k, 1, adjusted_max)
        full_mask = torch.cat([mask[:,0,:,:].unsqueeze(1) for i in range(values.shape[1])], dim=1)
        masked_image = values * full_mask

    def pairwise_distances(x):
        square_distances = torch.sum((x[:, None] - x) ** 2, dim=-1)
        distances = torch.sqrt(square_distances)
        return distances

    def average_distance(x):
        distances = pairwise_distances(x)
        num_points = x.shape[0]

        sum_distances = torch.sum(distances) / 2  # Divide by 2 to account for double counting

        return sum_distances

    def uncertainty(weights):
        return torch.mean(.25 - (.5 - weights) ** 2) * .0001

    losses = torch.zeros(K)
    means = torch.zeros((K, image.shape[1]))

    for k in range(K):
        image_copy = image.clone()
        classes_copy = classes.clone()
        classes_mask = (torch.argmax(classes_copy, dim=1) == k).float()

        #print(torch.stack([classes_mask for i in range(image_copy.shape[1])], dim=0).shape, image_copy.shape)

        # weighted_average(image, torch.cat([classes_copy[:,k,:,:].unsqueeze(1) for i in range(image_copy.shape[1])], dim=1)) #
        # var = weighted_average(classes_mask, image_copy, torch.cat([classes_copy[:,k,:,:].unsqueeze(1) for i in range(image_copy.shape[1])], dim=1))
        # uncert = uncertainty(classes_copy[:,k,:,:]) # uncertainty(torch.cat([classes_copy.unsqueeze(1) for i in range(image_copy.shape[1])], dim=1))
        means[k] = avg(classes_mask, image_copy,torch.cat([classes_copy[:,k,:,:].unsqueeze(1) for i in range(image_copy.shape[1])], dim=1))



        #losses[k] = var # torch.sum(torch.var(image_copy, dim=(2,3)))
    r = - average_distance(means)*1000
    return r #torch.mean(torch.pow(losses,1/2)) #/ torch.nonzero(losses).shape[0]


learning_rate = 0.001
model = FeatureNet()
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 10

# blur = torchvision.transforms.GaussianBlur(101)
last_improvement = 0
best_val = float("inf")

for epoch in tqdm(range(num_epochs)):
    running_loss = 0.0
    blur_loss = 0
    uncert_loss = 0
    few_loss = 0

    i = 0
    model.train()
    for data in tqdm(dl):
        # print(i)
        i += 1
        inputs = data
        inputs = inputs.to(device)

        optimizer.zero_grad()

        outputs, classes = model(inputs)

        # blur_classes = blur(classes)

        # # smoothness_loss = criterion(classes, blur_classes)*.005
        # up = 0#uncertianty_penalty(classes)
        # tp = 0# tv_regularization(classes)
        # fp = 0#few_channel_penalty(classes)
        # smoothness_loss = up + tp + fp
        # blur_loss += 0#tp.item()
        # uncert_loss += 0#up.item()
        # few_loss += 0#fp.item()
        # # smoothness_loss += torch.mean(ext_vals)
        #print(f"\t{inputs.shape}, {classes.shape}")
        # smoothness_loss = (Modified_Soft_Normalized_Cut_Loss(inputs.clone(), classes)).to(device)
        # #uc = uncertianty_penalty(classes).to(device)
        # #print(smoothness_loss.item(), uc.item())
        # #smoothness_loss += uc
        # smoothness_loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()
        

        # blur_loss += smoothness_loss.item()
        # outputs, classes = model(inputs)

        acc_loss = criterion(outputs, inputs)
        loss = acc_loss# + smoothness_loss * .01
        t_loss = loss #+ smoothness_loss
        acc_loss.backward()
        optimizer.step()
        running_loss += loss.item()
        #print(f"Batch results: {smoothness_loss.item()}, {loss.item()}, {running_loss}, {blur_loss}")


    # model.eval()
    # val_loss = 0
    # with torch.no_grad():
    #     for data in dl_val:
    #         inputs = data
    #         inputs = inputs.to(device)
    #
    #         outputs, classes = model(inputs)
    #
    #         # blur_classes = blur(classes)
    #
    #         # smoothness_loss = criterion(classes, blur_classes)*.005
    #         smoothness_loss = uncertianty_penalty(classes) + tv_regularization(classes) + few_channel_penalty(classes)
    #         # smoothness_loss += torch.mean(ext_vals)
    #
    #         # blur_loss += smoothness_loss.item()
    #
    #         acc_loss = criterion(outputs, inputs)
    #         loss = acc_loss + smoothness_loss
    #         val_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}]')

    print(f'\t\tReconstruction Loss: {(running_loss) / len(dl)}')
    print(f'\t\tSegmentation Loss: {(blur_loss) / len(dl)}')
    # print(f"\tVal Loss: {val_loss / len(dl_val)}")
    # if val_loss < best_val:
    #     best_val = val_loss
    #     last_improvement = epoch
    #     torch.save(model.state_dict(), "models/best_model.pickle")
    # if epoch - last_improvement >= 5:
    #     break

print('Finished Training')
model.eval()
with torch.no_grad():
    for i in range(22,30):
        print(i)
        example_image = ds_test.__getitem__(i)
    
        cv2.imwrite(f"images/item{i}green.tif", np.rint(example_image[0, :, :].numpy()*255).astype(np.uint8))
        cv2.imwrite(f"images/item{i}red.tif", np.rint(example_image[1, :, :].numpy()*255).astype(np.uint8))
        cv2.imwrite(f"images/item{i}nir.tif", np.rint(example_image[2, :, :].numpy()*255).astype(np.uint8))
        cv2.imwrite(f"images/item{i}re.tif", np.rint(example_image[3, :, :].numpy()*255).astype(np.uint8))
        #cv2.imwrite(f"initial_images/item{i}ndvi.tif", np.rint(example_image[4, :, :].numpy()*255).astype(np.uint8))
    
        ex = example_image.to(device)
        ex = torch.reshape(ex, (1, ex.shape[0], ex.shape[1], ex.shape[2]))
        pred, sing = model(ex)
    
        pred, sing = pred.detach().cpu(), sing.detach().cpu()  # sing.detach().cpu()
        pred = pred.squeeze()
        cv2.imwrite(f"images/item{i}PredG.tif", np.rint(pred[0, :, :].numpy() * 255).astype(np.uint8))
        cv2.imwrite(f"images/item{i}PredR.tif", np.rint(pred[1, :, :].numpy() * 255).astype(np.uint8))
        cv2.imwrite(f"images/item{i}PredNIR.tif", np.rint(pred[2, :, :].numpy() * 255).astype(np.uint8))
        cv2.imwrite(f"images/item{i}PredRE.tif", np.rint(pred[3, :, :].numpy() * 255).astype(np.uint8))


        cv2.imwrite(f"images/item{i}Class1.tif", np.rint(sing.squeeze()[0, :, :].numpy()*255).astype(np.uint8))
        cv2.imwrite(f"images/item{i}Class2.tif", np.rint(sing.squeeze()[1, :, :].numpy()*255).astype(np.uint8))
        cv2.imwrite(f"images/item{i}Class3.tif", np.rint(sing.squeeze()[2, :, :].numpy()*255).astype(np.uint8))
        cv2.imwrite(f"images/item{i}Class4.tif", np.rint(sing.squeeze()[3, :, :].numpy()*255).astype(np.uint8))
        cv2.imwrite(f"images/item{i}Class5.tif", np.rint(sing.squeeze()[4, :, :].numpy() * 255).astype(np.uint8))
        # cv2.imwrite(f"initial_images/item{i}Class5.tif", np.rint(sing.squeeze()[4, :, :].numpy()*255).astype(np.uint8))
        # cv2.imwrite(f"initial_images/item{i}Class6.tif", np.rint(sing.squeeze()[5, :, :].numpy() * 255).astype(np.uint8))
        # cv2.imwrite(f"initial_images/item{i}Class7.tif", np.rint(sing.squeeze()[6, :, :].numpy() * 255).astype(np.uint8))
        # cv2.imwrite(f"initial_images/item{i}Class8.tif", np.rint(sing.squeeze()[7, :, :].numpy() * 255).astype(np.uint8))
        # cv2.imwrite(f"initial_images/item{i}Class9.tif", np.rint(sing.squeeze()[8, :, :].numpy() * 255).astype(np.uint8))
        # cv2.imwrite(f"initial_images/item{i}Class10.tif", np.rint(sing.squeeze()[9, :, :].numpy() * 255).astype(np.uint8))



