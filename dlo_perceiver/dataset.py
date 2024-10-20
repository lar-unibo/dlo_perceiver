import torch, os, random, cv2, pickle
import numpy as np
from torch.utils.data import Dataset
import albumentations as aug

from dlo_perceiver.text_encoder import TextEncoder
from transformers import DistilBertTokenizer

COLORS = ["red", "green", "blue", "black", "white", "gray", "yellow", "orange", "purple", "brown", "striped"]
OBJECTS = ["cable", "rope"]
POSITIONS = ["top", "bottom"]


def get_transform(config):
    style = config.get("aug_style", 0)
    img_h = config.get("img_h", 360)
    img_w = config.get("img_w", 640)

    if style == 0:
        return aug.Compose(
            [
                aug.Flip(p=0.5),
                aug.FancyPCA(p=0.8),
                aug.GaussNoise(p=0.5),
                aug.ImageCompression(quality_lower=80, quality_upper=100, p=0.5),
                aug.RandomBrightnessContrast(contrast_limit=[0.1, 0.1], brightness_limit=[-0.1, 0.1]),
                aug.RandomSizedCrop(min_max_height=(300, 360), height=img_h, width=img_w, p=1.0),
            ],
            p=1,
        )
    elif style == 1:
        return aug.Compose(
            [
                aug.FancyPCA(always_apply=True),
                aug.HueSaturationValue(
                    hue_shift_limit=5, sat_shift_limit=[-50, 0], val_shift_limit=[-50, 0], always_apply=True
                ),
                aug.Flip(p=0.5),
                aug.AdvancedBlur(always_apply=True),
                aug.CropAndPad(percent=(-0.2, 0.4), pad_mode=cv2.BORDER_CONSTANT, pad_cval=0, keep_size=True),
                aug.GridDropout(always_apply=True),
                aug.Resize(height=config["img_h"], width=config["img_w"], always_apply=True),
            ],
            p=1,
        )
    elif style == 2:
        return aug.Compose(
            [
                aug.FancyPCA(always_apply=True),
                aug.HueSaturationValue(
                    hue_shift_limit=5, sat_shift_limit=[-50, 0], val_shift_limit=[-50, 0], always_apply=True
                ),
                aug.Flip(p=0.5),
                aug.AdvancedBlur(always_apply=True),
                aug.CropAndPad(percent=(-0.3, 0.4), pad_mode=cv2.BORDER_CONSTANT, pad_cval=0, keep_size=True),
                aug.Resize(height=config["img_h"], width=config["img_w"], always_apply=True),
            ],
            p=1,
        )
    elif style == 3:
        return aug.Compose(
            [
                aug.Perspective(scale=(0.01, 0.05), always_apply=True, pad_mode=cv2.BORDER_CONSTANT, pad_val=0),
                aug.FancyPCA(always_apply=True),
                aug.HueSaturationValue(
                    hue_shift_limit=5, sat_shift_limit=[-50, 50], val_shift_limit=[-50, 50], always_apply=True
                ),
                aug.Flip(p=0.5),
                aug.AdvancedBlur(always_apply=True),
                aug.GridDistortion(
                    num_steps=5, distort_limit=0.3, border_mode=cv2.BORDER_CONSTANT, value=0, always_apply=True
                ),
                aug.CropAndPad(percent=(-0.3, 0.4), pad_mode=cv2.BORDER_CONSTANT, pad_cval=0, keep_size=True),
                aug.Resize(height=config["img_h"], width=config["img_w"], always_apply=True),
            ],
            p=1,
        )
    elif style == 4:
        return aug.Compose(
            [
                aug.Perspective(scale=(0.01, 0.05), always_apply=True, pad_mode=cv2.BORDER_CONSTANT, pad_val=0),
                aug.FancyPCA(always_apply=True),
                aug.HueSaturationValue(
                    hue_shift_limit=5, sat_shift_limit=[-50, 50], val_shift_limit=[-50, 50], always_apply=True
                ),
                aug.Flip(p=0.5),
                aug.AdvancedBlur(always_apply=True),
                aug.GridDistortion(
                    num_steps=5, distort_limit=0.3, border_mode=cv2.BORDER_CONSTANT, value=0, always_apply=True
                ),
                aug.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=30,
                    border_mode=cv2.BORDER_CONSTANT,
                    always_apply=True,
                ),
                aug.CropAndPad(percent=(-0.3, 0.4), pad_mode=cv2.BORDER_CONSTANT, pad_cval=0, keep_size=True),
                aug.Resize(height=config["img_h"], width=config["img_w"], always_apply=True),
            ],
            p=1,
        )


def get_position(mask1, mask2):
    # Given the two masks identify the cable on top

    # find contours
    contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find the cable on top
    if len(contours1) == 1 and len(contours2) > 1:
        return "top", "bottom"
    elif len(contours2) == 1 and len(contours1) > 1:
        return "bottom", "top"
    else:
        # Not shure of what appening.... return None
        return None, None


def text_from_gt(data_gt, mask1, mask2):

    # Extract position either ['top', 'bottom'] / ['bottom', 'top'] / [None, None]
    p1, p2 = get_position(mask1, mask2)

    color1 = data_gt["curve_1"]["color"]
    color2 = data_gt["curve_2"]["color"]

    obj1 = data_gt["curve_1"].get("object_name", None)
    obj2 = data_gt["curve_2"].get("object_name", None)
    if obj1 is None or obj1 == "cable":
        obj1 = "cable"  # if torch.randint(0, 2, (1,)).item() == 0 else "wire"
    if obj2 is None or obj2 == "cable":
        obj2 = "cable"  # if torch.randint(0, 2, (1,)).item() == 0 else "wire"

    # check striped only for cables
    if obj1 == "rope" and color1 == "striped":
        color1 = "green"
    if obj2 == "rope" and color2 == "striped":
        color2 = "green"

    if obj1 is None or obj2 is None or color1 is None or color2 is None or p1 is None or p2 is None:
        return None

    return ({"obj": obj1, "color": color1, "pos": p1}, {"obj": obj2, "color": color2, "pos": p2})


class DloDataset(Dataset):
    def __init__(self, data_path, config=None):
        self.data_path = data_path
        self.data = sorted(os.listdir(data_path))
        self.config = config if config is not None else {}

        self.transform = get_transform(self.config)

        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.text_encoder = TextEncoder(model_name="distilbert-base-uncased")

        self.prob_use_color = self.config.get("dataset_prob_color", 0.0)
        self.prob_use_pos = self.config.get("dataset_prob_pos", 0.0)

    def __len__(self):
        return len(self.data)

    def pre_process(self, img):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        img = img.transpose((2, 0, 1))
        if img.max() > 1:
            img = img / 255
        return img

    def __getitem__(self, index):
        folder_name = self.data[index]

        # rgb
        img = cv2.imread(os.path.join(self.data_path, folder_name, "color.png"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # masks
        instances = cv2.imread(os.path.join(self.data_path, folder_name, "instances.png"), cv2.IMREAD_GRAYSCALE)

        # data_gt
        data_gt = pickle.load(open(os.path.join(self.data_path, folder_name, "gt.pkl"), "rb"))

        #############################
        masks_curve1 = (instances == 1).astype(np.uint8) * 255
        masks_curve2 = (instances == 2).astype(np.uint8) * 255
        mask_all = cv2.bitwise_or(masks_curve1, masks_curve2)  # Create mask with all the cables

        # Text
        prompt_gt = text_from_gt(data_gt["gt"], masks_curve1, masks_curve2)

        #############################
        use_color = random.random() < self.prob_use_color
        use_pos = random.random() < self.prob_use_pos

        if use_color and use_pos:
            prompt_1 = f"{prompt_gt[0]['obj']} {prompt_gt[0]['color']} {prompt_gt[0]['pos']}"
            prompt_2 = f"{prompt_gt[1]['obj']} {prompt_gt[1]['color']} {prompt_gt[1]['pos']}"
        elif use_color:
            prompt_1 = f"{prompt_gt[0]['obj']} {prompt_gt[0]['color']}"
            prompt_2 = f"{prompt_gt[1]['obj']} {prompt_gt[1]['color']}"
        elif use_pos:
            prompt_1 = f"{prompt_gt[0]['obj']} {prompt_gt[0]['pos']}"
            prompt_2 = f"{prompt_gt[1]['obj']} {prompt_gt[1]['pos']}"
        else:
            prompt_1 = f"{prompt_gt[0]['obj']}"
            prompt_2 = f"{prompt_gt[1]['obj']}"

        use_mask_1 = random.random() > 0.5
        if use_mask_1:
            mask = masks_curve1
            obj = prompt_gt[0]["obj"]
            color = prompt_gt[0]["color"]
            pos = prompt_gt[0]["pos"]
            prompt = prompt_1
            prompt_other = prompt_2
        else:
            mask = masks_curve2
            obj = prompt_gt[1]["obj"]
            color = prompt_gt[1]["color"]
            pos = prompt_gt[1]["pos"]
            prompt = prompt_2
            prompt_other = prompt_1

        if not use_color and not use_pos:
            if prompt_1 == prompt_2:
                mask = mask_all

        ##########################################
        # augmentations
        if self.transform is not None:
            augmented = self.transform(**{"image": img, "mask": mask})
            img_aug, mask_aug = augmented["image"], augmented["mask"]

        # HWC to CHW
        img_aug = self.pre_process(img_aug)
        mask_aug = self.pre_process(mask_aug)

        ######################
        # neg text
        neg_text = self.sample_neg_text(prompt_gt[0]["color"], prompt_gt[1]["color"])

        combined_texts = [prompt, neg_text]
        text_emb = self.tokenizer(combined_texts, return_tensors="pt", padding=True)["input_ids"]
        text_emb = self.text_encoder(text_emb)

        img_t = torch.from_numpy(img_aug).type(torch.FloatTensor)
        mask_t = torch.from_numpy(mask_aug).type(torch.FloatTensor)

        return img_t, combined_texts, text_emb, mask_t

    def sample_neg_texts(self, obj, color, pos, n=5, use_color=False, use_pos=False):

        obj_neg = OBJECTS[0] if obj == OBJECTS[1] else OBJECTS[1]
        pos_neg = POSITIONS[0] if pos == POSITIONS[1] else POSITIONS[1]
        colors_neg = [c for c in COLORS if c != color]
        colors = random.choices(colors_neg, k=n)

        ######
        # strong neg samples with same obj
        strong_neg_samples = []
        strong_neg_samples.append(f"{obj_neg} {color} {pos}")
        strong_neg_samples.append(f"{obj_neg} {color}")

        neg_samples = []
        neg_samples.append(f"{obj_neg}")
        neg_samples.append(f"{obj_neg} {pos_neg}")
        neg_samples.append(f"{obj_neg} {pos}")
        # obj + color
        for c in colors:
            neg_samples.append(f"{obj_neg} {c}")
        # obj + pos
        for c in colors:
            neg_samples.append(f"{obj_neg} {c} {pos_neg}")

        if use_color:
            for c in colors:
                neg_samples.append(f"{obj} {c}")
        elif use_color and use_pos:
            for c in colors:
                neg_samples.append(f"{obj} {c} {pos}")

        neg_samples = random.choices(neg_samples, k=n - 2)

        return neg_samples + strong_neg_samples

    def sample_neg_text(self, color1, color2, n=5):

        different_color = []
        for c in COLORS:
            if c != color1 and c != color2:
                different_color.append(c)

        obj = OBJECTS[0] if torch.randint(0, 2, (1,)).item() == 0 else OBJECTS[1]
        pos = POSITIONS[0] if torch.randint(0, 2, (1,)).item() == 0 else POSITIONS[1]
        color = random.choice(different_color)
        return f"{obj} {color} {pos}"


if __name__ == "__main__":
    dp = "/home/lar/dev24/DATASET_NEW3/val"

    data = DloDataset(
        data_path=dp,
        config={
            "img_h": 360,
            "img_w": 640,
            "aug_style": 4,
            "dataset_prob_color": 1.0,
            "dataset_prob_pos": 1.0,
        },
    )
    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)

    for i, batch in enumerate(loader):
        img, text, text_emb, mask = batch
        print(img.shape, text_emb.shape, mask.shape)

        img_np = img.squeeze(0).permute(1, 2, 0).numpy()
        mask_np = mask.squeeze(0).permute(1, 2, 0).numpy()
