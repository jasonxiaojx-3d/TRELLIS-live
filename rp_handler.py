print("jasonxiao: begin importing runpod")
import runpod
import time
import os
import logging
logging.info("jasonxiao: begin importing trellis required items")
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
logging.info("jasonxiao: trellis required items imported")
print("jasonxiao: begin loading trellis pipeline")
# Load a pipeline from a model folder or a Hugging Face model hub.
# pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
print("jasonxiao sanity check. RP handler loaded.")

def handler(event):
#   This function processes incoming requests to your Serverless endpoint.
#
#    Args:
#        event (dict): Contains the input data and request metadata
#       
#    Returns:
#       Any: The result to be returned to the client
    
    # Extract input data
    print(f"JX: Worker Start")
    input = event['input']
    
    prompt = input.get('prompt')  
    seconds = input.get('seconds', 0)  

    print(f"Received prompt: {prompt}")
    print(f"jasonxiao sanity check.")
    print(f"Sleeping for {seconds} seconds...")
    
    # You can replace this sleep call with your own Python code
    time.sleep(seconds)  
    
    return prompt 

# Start the Serverless function when the script is run
if __name__ == '__main__':
    print("jasonxiao sanity check. Main function loaded.")
    runpod.serverless.start({'handler': handler })