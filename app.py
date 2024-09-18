import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from PIL import Image
import streamlit as st
from torch import autocast

# Set device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_auth_token=True).to(torch_device)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(torch_device)
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", use_auth_token=True).to(torch_device)

scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

# Streamlit UI
st.title("Stable Diffusion Image Generator")
prompt = st.text_input("Enter your prompt:", value="A digital illustration of a steampunk computer laboratory with clockwork machines, 4k, detailed, trending in artstation, fantasy vivid colors")
num_inference_steps = st.slider("Number of inference steps", min_value=10, max_value=100, value=50)
guidance_scale = st.slider("Guidance scale", min_value=1.0, max_value=15.0, value=7.5)
submit = st.button("Generate Image")

if submit:
    with st.spinner("Generating image..."):
        # Generate the image based on the input prompt
        height = 512
        width = 768
        generator = torch.manual_seed(4)
        batch_size = 1

        # Tokenize and encode prompt
        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        with torch.no_grad():
            text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Set timesteps for the scheduler
        scheduler.set_timesteps(num_inference_steps)

        # Prep latent input
        latents = torch.randn((batch_size, unet.config.in_channels, height // 8, width // 8), generator=generator).to(torch_device)
        latents = latents * scheduler.sigmas[0]  # Scale latents to match the first noise level

        # Denoising loop
        with autocast("cuda"):
            for i, t in enumerate(scheduler.timesteps):
                latent_model_input = torch.cat([latents] * 2)
                sigma = scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

                # Predict noise
                with torch.no_grad():
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

                # Perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Update latents
                latents = scheduler.step(noise_pred, t, latents)["prev_sample"]

        # Decode latents with VAE
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample

        # Convert the image to a displayable format
        image = (image / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(img) for img in images]

        # Display the generated image in Streamlit
        st.image(pil_images[0], caption="Generated Image", use_column_width=True)

