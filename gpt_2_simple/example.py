import gpt_2_simple as gpt2
import os
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Disable eager execution 
disable_eager_execution()


gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
if gpus:
    try:
        
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                 gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6000)]
            )
    except RuntimeError as e:
        print(e)

model_name = "124M"
if not os.path.isdir(os.path.join("models", model_name)):
	print(f"Downloading {model_name} model...")
	gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/


# Start a TensorFlow session with GPU enabled
tf.compat.v1.reset_default_graph()  # Use the compat.v1 module
sess = gpt2.start_tf_sess()

# Load and fine-tune the GPT-2 model
file_name = "/mnt/c/Users/paul/github.com/pauldanielconway/personal-computer/home/Workstation/Python/resources/Personal_Training_Nutrition-Ocr_0.5.txt"
#file_name = "C:/Users/paul/github.com/pauldanielconway/personal-computer/home/Workstation/Python/resources/Personal_Training_Nutrition-Ocr_0.5.txt"

run_name = "run10"

gpt2.finetune(sess,
              dataset=file_name,
              model_name=model_name,
              only_train_transformer_layers=True,
              steps=1000,
              sample_length=1024,
              batch_size=1,
              learning_rate=0.00001,
              save_every=500,
              n_ctx=1024,
              n_embd=1024,
              n_vocab=50257,
              run_name=run_name)

# Generate text using the fine-tuned model
# gpt2.generate(sess)


# Example usage
question = "kcal per gram in glucose equals"
# Generate text and store it in a variable
generated_text = gpt2.generate(sess, run_name=run_name, prefix=question, length=100, return_as_list=True)[0]
# Print the generated text
print(generated_text)

