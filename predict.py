import os, torch, cv2, random
from tqdm import tqdm
import matplotlib.pyplot as plt

from dlo_perceiver.model_contrastive import DLOPerceiver
from dlo_perceiver.text_encoder import TextEncoder
from transformers import DistilBertTokenizer

PATH_TEST = "images"
CHECKPOINT_NAME = "dlo_perceiver.pt"

DEVICE = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE: ", DEVICE)


if __name__ == "__main__":
    project_path = os.path.dirname(os.path.realpath(__file__))
    print("project_path: ", project_path)
    model_path = os.path.join(project_path, CHECKPOINT_NAME)

    ###############
    # DATASET
    ###############

    path_imgs = PATH_TEST
    files = os.listdir(path_imgs)
    random.shuffle(files)
    print(f"Found {len(files)} test images!")

    ###############
    # MODEL
    ###############
    model_dict = torch.load(model_path)
    model_config = model_dict["config"]

    print("model_config: ", model_config)

    model_weights = model_dict["model"]

    model = DLOPerceiver(
        iterations=model_config["iterations"],
        n_latents=model_config["n_latents"],
        latent_dim=model_config["latent_dim"],
        depth=model_config["depth"],
        dropout=model_config["dropout"],
        img_encoder_type=model_config["img_encoder_type"],
    )

    model.load_state_dict(model_weights)
    model.to(device=DEVICE)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    text_encoder = TextEncoder(model_name="distilbert-base-uncased")

    ###############
    # EVAL
    ###############

    model.eval()
    total_val_loss = 0
    for i, file in enumerate(tqdm(files)):
        print(f"Processing image {i+1}/{len(files)}: {file}")

        img = cv2.imread(os.path.join(path_imgs, file), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (model_config["img_w"], model_config["img_h"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = img.copy()

        fig = plt.figure(figsize=(10, 5))
        plt.imshow(img_in)
        plt.tight_layout()
        plt.show(block=False)

        x = input("Enter the <object,color,top/bottom> of one DLO...")
        if x == "":
            break

        x = x.split(",")
        prompt = {"object": x[0], "color": x[1], "top_down": x[2]}

        ##############
        # prompts
        text = f"{prompt['object']} {prompt['color']} {prompt['top_down']}"

        print("text: ", text)
        texts = [text]

        text_emb = tokenizer(texts, return_tensors="pt", padding=True)["input_ids"]  #!!!!
        text_emb = text_encoder(text_emb).to(device=DEVICE)

        ###############
        img = img / 255.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(0).to(device=DEVICE)

        print("img.shape: ", img.shape)
        print("text_emb.shape: ", text_emb.shape)

        with torch.no_grad():
            pred, (x_img, x_conf, _) = model(img, text_emb)
        pred = pred.sigmoid()

        ##############
        # PLOT
        mask_out = pred.squeeze().detach().cpu().numpy()
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img_in)
        axs[0].set_title("Input Image")
        axs[0].axis("off")
        axs[1].imshow(mask_out)
        axs[1].set_title(texts[0])
        axs[1].axis("off")
        plt.tight_layout()

        plt.show()
