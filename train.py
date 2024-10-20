import os, wandb, torch, random
from tqdm import tqdm
import numpy as np
from torchvision.utils import make_grid

from dlo_perceiver.model_contrastive import DLOPerceiver
from dlo_perceiver.dataset import DloDataset
from dlo_perceiver.utils import PolyLR, warmup_learning_rate


WANDB_MODEL = "disabled"


DEVICE = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE: ", DEVICE)

CONFIG_WB = {
    "batch_size": 4,
    "lr": 1e-4,
    "epochs": 430,
    "seed": 42,
    "img_encoder_type": "resnet101",
    "dataset_prob_pos": 1.0,
    "dataset_prob_color": 1.0,
    "main_path": "path_here",
    "train_folder": "train",
    "val_folder": "val",
}

#############################################################

CONFIG = {
    # Perceiver Params
    "iterations": 1,
    "n_latents": 256,
    "latent_dim": 256,
    "depth": 3,
    "dropout": 0.05,
    "decoder_dropout": 0.05,
    ########
    "freq_validation_per_epoch": 2,
    "warmup_steps": 500,
    "warmup_lr": 1e-8,
    ########
    "img_h": 360,
    "img_w": 640,
    "aug_style": 2,
    "noise_aug": True,
}
CONFIG.update(CONFIG_WB)


def val(model, val_loader, criterion, global_step, log_img=True):
    ###############
    # VAL
    model.eval()
    total_val_loss, total_val_pred_loss, total_val_contrastive_loss = 0, 0, 0
    for i, batch in enumerate(tqdm(val_loader)):

        img, _, text_emb, label_mask = batch

        img = img.to(device=DEVICE)
        text_emb = text_emb.to(device=DEVICE)
        label_mask = label_mask.to(device=DEVICE)

        # pred
        with torch.no_grad():
            pred, (x0_conf, x1_conf, x2_conf) = model(img, text_emb, training=True)

            loss_pred = criterion(pred, label_mask)
            loss_contrastive = criterion_contrastive(x0_conf, x1_conf, x2_conf)
            loss = loss_pred + loss_contrastive

        total_val_loss += loss.item()
        total_val_pred_loss += loss_pred.item()
        total_val_contrastive_loss += loss_contrastive.item()

        if log_img:
            wandb.log({"val_images_pred": wandb.Image(make_grid(pred).to(torch.float))}, step=global_step)
            wandb.log({"val_images_gt": wandb.Image(make_grid(label_mask).to(torch.float))}, step=global_step)
            log_img = not log_img

    # total_val_conf_loss /= len(val_loader)
    total_val_pred_loss /= len(val_loader)
    total_val_loss /= len(val_loader)
    total_val_contrastive_loss /= len(val_loader)
    wandb.log(
        {
            "val_loss": total_val_loss,
            "val_loss_pred": total_val_pred_loss,
            "val_loss_contrastive": total_val_contrastive_loss,
        },
        step=global_step,
    )
    return total_val_loss


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    criterion_contrastive,
    scheduler,
    checkpoint_path,
    config,
):

    ###############
    # TRAINING
    ###############
    run_name = wandb.run.name + f"_{CONFIG['img_encoder_type']}"

    global_step = 0
    best_val_loss = 10e8
    best_model = None
    n_train = len(train_loader)
    total_val_loss, total_train_loss = 0, 0
    for epoch in range(CONFIG["epochs"]):

        # TRAIN
        model.train()
        total_train_loss = 0
        for i, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()

            ############################################
            # learning rate warmup
            warmup_learning_rate(optimizer, global_step, config)
            ############################################

            img, _, text_emb, label_mask = batch

            img = img.to(device=DEVICE)
            text_emb = text_emb.to(device=DEVICE)
            label_mask = label_mask.to(device=DEVICE)

            # pred
            pred, (x0_conf, x1_conf, x2_conf) = model(img, text_emb, training=True)

            # loss
            loss_pred = criterion(pred, label_mask)
            loss_contrastive = criterion_contrastive(x0_conf, x1_conf, x2_conf)

            loss = loss_pred + loss_contrastive
            total_train_loss += loss.item()

            # backward pass
            loss.backward()

            # optimize
            optimizer.step()

            # log
            wandb.log(
                {
                    "train_loss": loss.item(),
                    "train_loss_pred": loss_pred.item(),
                    "train_loss_contrastive": loss_contrastive.item(),
                }
            )

            # update global step
            global_step += 1

            # VALIDATION round
            if global_step % (n_train // config["freq_validation_per_epoch"]) == 0:
                total_val_loss = val(model, val_loader, criterion, global_step)
                wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=global_step)

        scheduler.step()

        total_train_loss = total_train_loss / len(train_loader)

        # Print
        print(f"Epoch: {epoch} | Train Loss: {total_train_loss} | Val Loss: {total_val_loss}")

        # Save Best Model
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            best_model = model.state_dict()
            state = {"model": best_model, "config": CONFIG}
            torch.save(state, os.path.join(checkpoint_path, f"{run_name}.pt"))
            print("new best model saved!")

        # Save Last Model
        state = {"model": model.state_dict(), "config": CONFIG}
        torch.save(state, os.path.join(checkpoint_path, f"{run_name}_LAST.pt"))


if __name__ == "__main__":
    # project root
    project_path = os.path.dirname(os.path.realpath(__file__))
    print("project_path: ", project_path)
    checkpoint_path = os.path.join(project_path, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)

    MAIN_PATH = CONFIG["main_path"]
    dataset_path_train = os.path.join(MAIN_PATH, CONFIG["train_folder"])
    dataset_path_val = os.path.join(MAIN_PATH, CONFIG["val_folder"])

    print("dataset_path_train: ", dataset_path_train)
    print("dataset_path_val: ", dataset_path_val)

    # Set seed
    torch.manual_seed(CONFIG["seed"])
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    #################################
    # DATASET
    dataset_train = DloDataset(data_path=dataset_path_train, config=CONFIG)
    dataset_val = DloDataset(data_path=dataset_path_val, config=CONFIG)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=CONFIG["batch_size"], shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=CONFIG["batch_size"], shuffle=False, drop_last=True
    )

    ###############
    # MODEL
    ###############
    model = DLOPerceiver(
        iterations=CONFIG["iterations"],
        n_latents=CONFIG["n_latents"],
        latent_dim=CONFIG["latent_dim"],
        depth=CONFIG["depth"],
        dropout=CONFIG["dropout"],
        img_encoder_type=CONFIG["img_encoder_type"],
        noise_aug=CONFIG["noise_aug"],
    )
    model.to(device=DEVICE)

    ################################################

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    criterion_contrastive = torch.nn.TripletMarginWithDistanceLoss()
    scheduler = PolyLR(optimizer, CONFIG["epochs"], power=0.97, min_lr=1e-7)

    # Start WANDB
    wandb.init(config=CONFIG, project="dlo-perceiver", mode=WANDB_MODEL)
    train(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        criterion_contrastive,
        scheduler,
        checkpoint_path,
        CONFIG,
    )
    wandb.finish()

    print("Train completed!\n\n")
