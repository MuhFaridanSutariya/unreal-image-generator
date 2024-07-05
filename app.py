import replicate
import os

os.environ['REPLICATE_API_TOKEN'] = ''

model_1 = replicate.models.get("tstramer/midjourney-diffusion")
version_1 = model_1.versions.get("436b051ebd8f68d23e83d22de5e198e0995357afef113768c20f0b6fcef23c8b")

def generate_image(persona):
    prompt=f"A close up photo for person with following characteristic: {persona}, front face, close up headshot photo\
    shot in 8K, front look, face detailed, highly detailed, headshot potrait, close up headshot, close up potrait shot, \
    natural lighting, headshot photo, headshot photograph, 8k quality headshot, headshot profile picture, official photo"
    
    #https://replicate.com/tstramer/midjourney-diffusion/versions/436b051ebd8f68d23e83d22de5e198e0995357afef113768c20f0b6fcef23c8b#input
    inputs = {
        # Input prompt
        'prompt': prompt,

        # Specify things to not see in the output
        # 'negative_prompt': ...,

        # Width of output image. Maximum size is 1024x768 or 768x1024 because
        # of memory limits
        'width': 768,

        # Height of output image. Maximum size is 1024x768 or 768x1024 because
        # of memory limits
        'height': 768,

        # Prompt strength when using init image. 1.0 corresponds to full
        # destruction of information in init image
        'prompt_strength': 0.8,

        # Number of images to output.
        # Range: 1 to 4
        'num_outputs': 1,

        # Number of denoising steps
        # Range: 1 to 500
        'num_inference_steps': 50,

        # Scale for classifier-free guidance
        # Range: 1 to 20
        'guidance_scale': 7.5,

        # Choose a scheduler.
        'scheduler': "DPMSolverMultistep",

        # Random seed. Leave blank to randomize the seed
        # 'seed': ...,
    }
    
    # https://replicate.com/tstramer/midjourney-diffusion/versions/436b051ebd8f68d23e83d22de5e198e0995357afef113768c20f0b6fcef23c8b#output-schema
    output = version_1.predict(**inputs)
    return output[0]
