from src.pipeline_difix import DifixPipeline
from diffusers.utils import load_image
import time

pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
pipe.to("cuda")

# 记录开始时间
start_time = time.time()
input_image = load_image("mydata/images/test.png")
prompt = "remove degradation"

output_image = pipe(prompt, image=input_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
output_image.save("mydata/output/example_output.png")
# 记录结束时间
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")