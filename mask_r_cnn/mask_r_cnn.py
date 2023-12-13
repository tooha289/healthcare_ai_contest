import warnings
warnings.filterwarnings('ignore')

import os
import time
import random
import collections

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
fix_all_seeds(2021)
# 경로
TRAIN_CSV = '../sartorius-cell-instance-segmentation/train.csv'
TRAIN_PATH = '../sartorius-cell-instance-segmentation/train'
TEST_PATH = '../sartorius-cell-instance-segmentation//test'

# 원본이미지 크기
WIDTH = 704
HEIGHT = 520

# True 인 경우 다른 조건을 주기 위함
TEST = False    
NORMALIZE = False
USE_SCHEDULER = False

# 데이터 전처리
resnet_mean = (0.485, 0.456, 0.406)
resnet_std = (0.229, 0.224, 0.225)

# model parameters
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
n_epochs = 10
batch_size = 16
momentum = 0.9
lr = 0.001
weight_decay = 0.0005

# mask rcnn 에서는 0.5를 기준으로 mask가 있는지 없는지 판단
mask_threshold = 0.5

# 이미지 당 detection 최대 갯수
# 세포 이미지의 경우 500대 단위를 많이 사용하는 편으로, 이미지에 따라 수정
box_detections_per_img = 539

# 겹치는 부분 score 기준
# 0 또는 0.5부터 가장 좋은 결과를 가질 때까지 수정
min_score = 0.5


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
# 이미지 플립
class VerticalFlip:
    def __init__(self, prob):
        self.prob = prob
        
    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-2)
            bbox = target['boxes']
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]]
            target['boxes'] = bbox
            target['masks'] = target['masks'].flip(-2)
        return image, target
    
class HorizontalFlip:
    def __init__(self, prob):
        self.prob = prob
        
    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target['boxes']
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target['boxes'] = bbox
            target['masks'] = target['masks'].flip(-1)
        return image, target
# 데이터 처리
class Normalize:
    def __call__(self, image, target):
        image = F.normalize(image, RESNET_MEAN, RESNET_STD)
        return image, target
    
class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
    
def get_transform(train):
    transforms = [ToTensor()]
    if NORMALIZE:
        transforms.append(Normalize())
    
    # Data augmentation for train
    if train: 
        transforms.append(HorizontalFlip(0.5))
        transforms.append(VerticalFlip(0.5))

    return Compose(transforms)
# 마스크 표기하는 함수
# shape: (height, width)
# return 0:background, 1:mask

def rle_decode(annotation, shape, color=1):
    s = annotation.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = color
    return img.reshape(shape)


class CellDataset(Dataset):
    def __init__(self, image_dir, df, transforms=None, resize=False):
        self.transforms = transforms
        self.image_dir = image_dir
        self.df = df
        self.should_resize = resize is not False
        
        if self.should_resize:
            self.height = int(HEIGHT * resize)
            self.width = int(WIDTH * resize)
        else:
            self.height = HEIGHT
            self.width = WIDTH
        
        self.image_info = collections.defaultdict(dict)
        temp_df = self.df.groupby('id')['annotation'].agg(lambda x: list(x)).reset_index()
        
        for index, row in temp_df.iterrows():
            self.image_info[index] = {'image_id': row['id'], 
                                      'image_path': os.path.join(self.image_dir, row['id']+'.png'),
                                      'annotations': row['annotation']}
            
    def get_box(self, a_mask):
        # 주어진 mask로부터 bbox 확보
        pos = np.where(a_mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        return [xmin, ymin, xmax, ymax]

    def __getitem__(self, idx):
        img_path = self.image_info[idx]['image_path']
        img = Image.open(img_path).convert('RGB')
        
        if self.should_resize:
            img = img.resize((self.width, self.height), resample=Image.BILINEAR)

        info = self.image_info[idx]

        n_objects = len(info['annotations'])
        masks = np.zeros((len(info['annotations']), self.height, self.width), dtype=np.uint8)
        
        # bbox 좌표 얻기
        boxes = []
        for i, annotation in enumerate(info['annotations']):
            a_mask = rle_decode(annotation, (HEIGHT, WIDTH))
            a_mask = Image.fromarray(a_mask)

            if self.should_resize:
                a_mask = a_mask.resize((self.width, self.height), resample=Image.BILINEAR)

            a_mask = np.array(a_mask) > 0
            masks[i, :, :] = a_mask
            boxes.append(self.get_box(a_mask))

        # dummy lables
        labels = [1 for _ in range(n_objects)]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((n_objects,), dtype=torch.int64)

        target = {'boxes': boxes,
                  'labels': labels,
                  'masks': masks,
                  'image_id': image_id,
                  'area': area,
                  'iscrowd': iscrowd}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.image_info)
df_train = pd.read_csv(TRAIN_CSV, nrows=5000 if TEST else None)
ds_train = CellDataset(TRAIN_PATH, df_train, resize=False, transforms=get_transform(train=True))
dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))


def get_model():
    NUM_CLASSES = 2
    
    if NORMALIZE:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,
                                                                   box_detections_per_img=box_detections_per_img,
                                                                   image_mean=resnet_mean,
                                                                   image_std=resnet_std)
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,
                                                                   box_detections_per_img=box_detections_per_img)
    
    # pretrained 모델의 필요한 부분을 가져오고 새로운 학습을 위해서 해서 설정해야 하는 부분 설정해주기
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    
    # get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # replace mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, NUM_CLASSES)
    
    return model
model = get_model()
model.to(device)

for param in model.parameters():
    param.requires_grad = True
    
model.train()
# pretrained 모델의 구조에서 roi head에 있는 box predictor와 mask predictor의 output이 바뀐걸 확인할 수 있음



params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
n_batches = len(dl_train)
for epoch in range(n_epochs):
    
    time_start = time.time()
    loss_accum = 0.0
    loss_mask_accum = 0.0
    
    for batch_idx, (images, targets) in enumerate(dl_train, 1):
        
        # predic
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # logging
        loss_mask = loss_dict['loss_mask'].item()
        loss_accum += loss.item()
        loss_mask_accum += loss_mask
        
        # if batch_idx % 10 == 0:
        #     print(f'Batch {batch_idx:3d}/{n_batches:3d}  Batch Train Loss {loss.item():7.3f}  Mask Only Loss {loss_mask:7.3f}')
            
    if USE_SCHEDULER:
        lr_scheduler.step()
        
    train_loss = loss_accum/n_batches
    train_loss_mask = loss_mask_accum/n_batches
    
    take_time = time.time()-time_start
    
    # torch.save(model.state_dict(), f'model_{epoch}.bin')
    prefix = f'Epoch {epoch:2d}/{n_epochs:2d}'
    print(f'{prefix} -- Train Loss {train_loss:.3f} -- Train Mask Only Loss {train_loss_mask:.3f} -- Take time {take_time:.0f}sec')
# train 시각화 함수
def analyze_train_sample(model, ds_train, sample_index=10):
    
    # sample image
    img, targets = ds_train[sample_index]
    plt.imshow(img.numpy().transpose((1, 2, 0)))
    plt.title('Sample Image')
    plt.show()
    
    # ground truth
    masks = np.zeros((HEIGHT, WIDTH))
    for mask in targets['masks']:
        masks = np.logical_or(masks, mask)
        
    plt.imshow(img.numpy().transpose((1, 2, 0)))
    plt.imshow(masks, alpha=0.3)
    plt.title('Ground Truth')
    plt.show()
    
    # pred
    model.eval()
    with torch.no_grad():
        preds = model([img.to(device)])[0]
        
    plt.imshow(img.cpu().numpy().transpose((1, 2, 0)))
    all_preds_masks = np.zeros((HEIGHT, WIDTH))
    for mask in preds['masks'].cpu().detach().numpy():
        all_preds_masks = np.logical_or(all_preds_masks, mask[0] > mask_threshold)
    plt.imshow(all_preds_masks, alpha=0.4)
    plt.title('Predictions')
    plt.show
    
    return preds
    
sample_vi = analyze_train_sample(model, ds_train, 20)

# get mask 
def rle_encoding(x):
    dots = np.where(x.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    
    for b in dots:
        if (b > prev+1):
            run_lengths.extend((b+1, 0))
            run_lengths[-1] += 1
            prev = b
            
    return ' '.join(map(str, run_lengths))

# 겹치는 부분 지우기
def remove_overlapping_pixels(mask, other_masks):
    for other_mask in other_masks:
        if np.sum(np.logical_and(mask, other_mask)) > 0:
            mask[np.logical_and(mask, other_mask)] = 0
            
    return mask
model.eval()
submission = []

for sample in ds_test:
    img = sample['image']
    image_id = sample['image_id']
    with torch.no_grad():
        result = model([img.to(device)])[0]
        
    previous_masks = []
    for i, mask in enumerate(result['masks']):
        score = result['scores'][i].cpu().item()
        if score < min_score:
            continue
            
        # 가장 연관성 높은 mask만 남기기
        mask = mask.cpu().numpy()
        binary_mask = mask > mask_threshold
        binary_mask = remove_overlapping_pixels(binary_mask, previous_masks)
        previous_masks.append(binary_mask)
        rle = rle_encoding(binary_mask)
        submission.append((image_id, rle))
        
    # 이미지에 대해서 rle가 생성되지 않으면 빈 prediction 추가
    all_images_ids = [image_id for image_id, rle in submission]
    if image_id not in all_images_ids:
        submission.append((image_id, ''))
# 제출
df_sub = pd.DataFrame(submission, columns=['id', 'pred'])
df_sub.to_csv('../data/paper_review/maskrcnn_cell/submission.csv', index=False)
df_sub.head()


# min_score 값 조절
min_score = 0.43

def visualized_test(ds_test, sample_index):
    
    # test sample
    img = ds_test[sample_index]['image']
    image_id = ds_test[sample_index]['image_id']
    plt.imshow(img.numpy().transpose((1, 2, 0)))
    plt.title('Test Sample Image')
    plt.show()
    
    # test predict
    model.eval()
    with torch.no_grad():
        result = model([img.to(device)])[0]
        
    plt.imshow(img.numpy().transpose((1, 2, 0)))
    all_preds_masks = np.zeros((HEIGHT, WIDTH))
    
    previous_masks = []
    for i, mask in enumerate(result['masks']):
        score = result['scores'][i].cpu().item()
        if score < min_score:
            continue
            
        # 가장 연관성 높은 mask만 남기기
        mask = mask.cpu().numpy()
        binary_mask = mask > mask_threshold
        binary_mask = remove_overlapping_pixels(binary_mask, previous_masks)
        previous_masks.append(binary_mask)
        
        for mask in previous_masks:
            all_preds_masks = np.logical_or(all_preds_masks, mask[0] > mask_threshold)
        
    plt.imshow(all_preds_masks, alpha=0.4)
    plt.title('Test Pred')
    plt.show()
visualized_test(ds_test, 1)
# visualized_test(ds_test, 2)
# visualized_test(ds_test, 3)