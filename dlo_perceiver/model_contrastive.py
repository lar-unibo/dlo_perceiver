import torch
from einops import repeat
from dlo_perceiver.attention_rot import CrossAttentionLayer, Attention
from dlo_perceiver.image_encoder import ResNetEncoder


class DLOPerceiver(torch.nn.Module):
    def __init__(
        self,
        iterations=3,
        n_latents=256,
        latent_dim=256,
        cross_heads=4,
        cross_dim_head=4,
        latent_heads=4,
        latent_dim_head=4,
        dropout=0.0,
        depth=3,
        img_encoder_type="resnet50",
        noise_aug=False,
    ):
        super().__init__()

        self.iterations = iterations
        self.img_encoder_type = img_encoder_type
        self.noise_aug = noise_aug

        ############ ENCODERS ############
        print(f"Using image encoder: {img_encoder_type}")
        if img_encoder_type == "resnet50":
            self.image_encoder = ResNetEncoder("resnet50", latent_dim)
        elif img_encoder_type == "resnet101":
            self.image_encoder = ResNetEncoder("resnet101", latent_dim)
        elif img_encoder_type == "swint":
            self.image_encoder = SwinEncoder("swint", latent_dim)
        elif img_encoder_type == "swins":
            self.image_encoder = SwinEncoder("swins", latent_dim)
        elif img_encoder_type == "swinb":
            self.image_encoder = SwinEncoder("swinb", latent_dim)
        else:
            raise ValueError(f"Invalid encoder type: {img_encoder_type}")

        self.text_proj = torch.nn.Linear(768, latent_dim)

        ######## Perceiver
        self.latents = torch.nn.Parameter(torch.normal(0, 0.2, (n_latents, latent_dim)))
        self.latents_conf = torch.nn.Parameter(torch.normal(0, 0.2, (1, latent_dim)))

        self.ins_projector = torch.nn.Linear(latent_dim + latent_dim, latent_dim)

        self.cross_attention = CrossAttentionLayer(
            dim=latent_dim,
            depth=depth,
            iterations=iterations,
            cross_heads=cross_heads,
            latent_heads=latent_heads,
            cross_dim_head=cross_dim_head,
            latent_dim_head=latent_dim_head,
            dropout=dropout,
        )

        # decoder cross attention
        self.decoder_cross_attn = Attention(latent_dim, heads=cross_heads, dim_head=cross_dim_head, dropout=dropout)

        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(latent_dim, latent_dim, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(latent_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(latent_dim, 1, 1),
        )

        #################
        self.img_contrastive_embed = torch.nn.Sequential(
            torch.nn.Conv2d(latent_dim, latent_dim // 2, 3, padding=1, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(latent_dim // 2, latent_dim // 4, 3, padding=1, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(latent_dim // 4, latent_dim // 4, 3, padding=1, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(latent_dim // 4, latent_dim // 4, 3, padding=1, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
        )

        self.img_contrastive_embed2 = torch.nn.Linear(latent_dim // 4 * 5 * 10, latent_dim)
        self.text_contrastive_embed = torch.nn.Linear(latent_dim, latent_dim)

    def forward(self, rgb, text_emb, training=False):
        """
        rgb: Bx3xHxW
        text: BxN
        """
        #############################
        # Image processing
        input_shape = rgb.shape[-2:]

        # Image embedding via encoder
        img_embeddings = self.image_encoder(rgb)  # B x latent_dim x H/4 x W/4
        B, F, H, W = img_embeddings.shape
        ie = img_embeddings.view(B, F, -1).permute(0, 2, 1)  # Flatten the last two dimensions for each batch element

        if self.noise_aug:
            ie_noise = torch.randn(ie.shape) * (0.1**0.5)  # guassian noise zero mean and 0.1 variance
            ie = ie + ie_noise.to(ie.device)

        #############################
        # Text embedding projection
        p = self.text_proj(text_emb)  # B x 8 x latent_dim

        print("image proj ", ie.shape)
        print("text proj ", p.shape)

        if training:
            x0_latents = self.forward_perceiver(ie, p[:, 0, :].unsqueeze(1))
            x_img = self.img_contrastive_embed(img_embeddings)
            x_img = self.img_contrastive_embed2(x_img.view(B, -1))
            x_pos_text = self.text_contrastive_embed(p[:, 0, :])
            x_neg_text = self.text_contrastive_embed(p[:, 1, :])

            x_img = x_img / x_img.norm(dim=-1, keepdim=True)
            x_pos_text = x_pos_text / x_pos_text.norm(dim=-1, keepdim=True)
            x_neg_text = x_neg_text / x_neg_text.norm(dim=-1, keepdim=True)

        else:
            x0_latents = self.forward_perceiver(ie, p)
            print("xo latents ", x0_latents.shape)
            x_img = self.img_contrastive_embed(img_embeddings)
            x_img = self.img_contrastive_embed2(x_img.view(B, -1))
            x_pos_text = self.text_contrastive_embed(p)
            x_neg_text = None

            x_img = x_img / x_img.norm(dim=-1, keepdim=True)
            x_pos_text = x_pos_text / x_pos_text.norm(dim=-1, keepdim=True)

        # classification head starting from the latents
        img_latents = x0_latents.permute(0, 2, 1).view(B, F, H, W)  # B x latent_dim x H x W
        out_mask = self.classifier(img_latents)
        out_mask = torch.nn.functional.interpolate(out_mask, size=input_shape, mode="bilinear", align_corners=False)

        return out_mask, (x_img, x_pos_text, x_neg_text)

    def forward_perceiver(self, ie, p):

        # CAT Vision + Language
        ins = self.ins_projector(torch.cat((ie, p.repeat(1, ie.shape[1], 1)), dim=-1))  # B x (H/4 * W/4) x latent_dim

        print("cat ", ins.shape)

        # Perceiver
        # prepare latents based on the batch size
        x = repeat(self.latents, "n d -> b n d", b=ie.shape[0])  # B x n_latents x latent_dim

        print("latents ", self.latents.shape, x.shape)

        # Cross attention between latents query and [image+text]
        x = self.cross_attention(x, context=ins)

        print(x.shape)

        # Decoder of the latent image embedded
        x_latents = self.decoder_cross_attn(ins, context=x)

        print(x_latents.shape)

        return x_latents


if __name__ == "__main__":

    img = torch.randn(8, 3, 360, 640)
    text = torch.randn(8, 1, 768)

    model = DLOPerceiver()

    x, _ = model(img, text)

    print("input ", img.shape, text.shape)
    print("out ", x.shape)
