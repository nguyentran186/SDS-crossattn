import torch
from diffusers import StableDiffusionXLInpaintPipeline
from guidance.utils import retrieve_timesteps
from torch.cuda.amp import custom_bwd, custom_fwd
from PIL import Image

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

# Load the inpainting model
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,  # Use float16 for efficiency
).to("cuda")  # Move model to GPU if available 

def train_step(prompt: str = None,
               image = None,
               mask_image = None,
               guidance_scale: float = 7.5,
               strength: float = 0.9999,
               num_inference_steps: int = 50,
               timesteps: int = None,
               eta: float = 0.0,
               do_classifier_free_guidance: bool = True
               ):
    # 0. Default height and width to unet
    height = pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = pipe.unet.config.sample_size * pipe.vae_scale_factor
    
    # 1. Check inputs
        #skip
    
    # 2. Define call parameters
    device = pipe.device
    batch_size = 1
    
    with torch.no_grad():
        # 3. Encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

    # 4. set timesteps
    def denoising_value_valid(dnv):
        return isinstance(dnv, float) and 0 < dnv < 1
    
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler, num_inference_steps, device, timesteps, None
    )
    
    timesteps, num_inference_steps = pipe.get_timesteps(
        num_inference_steps,
        strength,
        device,
        denoising_start=None,
    )
    
    latent_timestep = timesteps[:1].repeat(batch_size)
    
    # 5. Preprocess mask and image
    is_strength_max = strength == 1.0
    
    crops_coords = None
    resize_mode = "default"
    
    original_image = image
    init_image = pipe.image_processor.preprocess(
        image, height=height, width=width, crops_coords=crops_coords, resize_mode=resize_mode
    )
    
    init_image = init_image.to(dtype=torch.float32)
    
    with torch.no_grad():
        if mask_image is None:
            mask_image = torch.zeros((1, 1, height, width), device=device)
        
        mask = pipe.mask_processor.preprocess(
            mask_image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
        )
    
    masked_image = init_image * (mask < 0.5)
    
    
    # 6. Prepare latent variables
    num_channels_latents = pipe.vae.config.latent_channels
    num_channels_unet = pipe.unet.config.in_channels
    
    generator = torch.Generator().manual_seed(42)
    
    latents_outputs = pipe.prepare_latents(
        batch_size,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        None,
        image=init_image,
        timestep=latent_timestep,
        is_strength_max=is_strength_max,
        add_noise=False,
        return_noise=True,
        return_image_latents=False,
    )
    
    latents, noise = latents_outputs
    
    with torch.no_grad():
    # 7. Prepare mask latent variables
        mask, masked_image_latents = pipe.prepare_mask_latents(
            mask,
            masked_image,
            batch_size,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            do_classifier_free_guidance,
        )
    
    # 8. Check that sizes of mask, masked image and latents match
    if num_channels_unet == 9:
        # default case for runwayml/stable-diffusion-inpainting
        num_channels_mask = mask.shape[1]
        num_channels_masked_image = masked_image_latents.shape[1]
        if num_channels_latents + num_channels_mask + num_channels_masked_image != pipe.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {pipe.unet.config} expects"
                f" {pipe.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                " `pipeline.unet` or your `mask_image` or `image` input."
            )
    elif num_channels_unet != 4:
        raise ValueError(
            f"The unet {pipe.unet.__class__} should have either 4 or 9 input channels, not {pipe.unet.config.in_channels}."
        )
    # 8.1 Prepare extra step kwargs.
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)
    
    # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    height, width = latents.shape[-2:]
    height = height * pipe.vae_scale_factor
    width = width * pipe.vae_scale_factor

    original_size = (height, width)
    target_size = (height, width)
    
    # 10. Prepare added time ids & embeddings
    negative_original_size = original_size
    negative_target_size = target_size

    add_text_embeds = pooled_prompt_embeds
    if pipe.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim

    add_time_ids, add_neg_time_ids = pipe._get_add_time_ids(
        original_size,
        (0, 0),
        target_size,
        6.0,
        2.5,
        negative_original_size,
        (0, 0),
        negative_target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )
    add_time_ids = add_time_ids.repeat(batch_size, 1)

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_neg_time_ids = add_neg_time_ids.repeat(batch_size, 1)
        add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device)

    # 11. Denoising loop
    # expand the latents if we are doing classifier free guidance
    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    
    # concat latents, mask, masked_image_latents in the channel dimension
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, latent_timestep)
    
    if num_channels_unet == 9:
        latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1) 

    # predict the noise residual
    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

    with torch.no_grad():
        noise_pred = pipe.unet(
            latent_model_input,
            latent_timestep,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=None,
            cross_attention_kwargs=None,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
    w = lambda alphas: (((1 - alphas) / alphas) ** 0.5)  
    
    alphas = pipe.scheduler.alphas_cumprod.to(device)
    
    grad = w(alphas[latent_timestep.to(torch.int)]) * (noise_pred - noise)
    
    grad = torch.nan_to_num(grad)
    # loss = SpecifyGradient.apply(latents, grad)
    loss = latents.sum()

    return loss

    
