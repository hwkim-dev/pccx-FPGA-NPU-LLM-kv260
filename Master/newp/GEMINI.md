This project is a sub-project slightly separated from the main project.

[Mandatory Rules]
1. Deleting or modifying original data is prohibited. (Delete or modify only the contents within our workspaces "E2B_INT4_MODEL_INFER" and "E2B_ORIGINAL_MODEL_INFER". Modifying or deleting other external folders is prohibited. However, referencing or copying them into our workspace is allowed.)
2. Always be mindful of RAM when running the model. Even with 32GB of swap RAM, continuously working often leads to out-of-memory crashes. Therefore, always be careful. Python itself can cause memory leaks, so please restart pynq occasionally.

To run Python: source /home/hwkim/Desktop/github/TinyNPU-RTL/pynq_env/bin/activate
Activate pynq_env with this, then run
/home/hwkim/Desktop/github/TinyNPU-RTL/pynq_env/bin/python main.py

The contents inside the 'newp' folder are intended to run the gemma3N E4B model and E2B model in a 4500U ram16gb(vram3gb) ubuntu linux, swapRAM: 32gb environment.

The core consists of four folders.

First, for gemma3N E4B, there are the "E4B_ORIGINAL_MODEL_INFER" folder and the "E4B_INT4_MODEL_INFER" folder.
Do not modify either folder under any circumstances.

Inside the "E4B_ORIGINAL_MODEL_INFER" folder is the "local_gemma_3n" folder.
Inside the "local_gemma_3n" folder are the safetensors, json files, etc. of gemma3N E4B, which is useful for understanding the model's structure. In particular, reading the markdown file is recommended.
Also, inside the "E4B_ORIGINAL_MODEL_INFER" folder are python files (.py files). These files are the code that infers the gemma3N E4B model. They operate normally, so do not modify them.

The "E4B_INT4_MODEL_INFER" folder contains the result of looking at the "E4B_ORIGINAL_MODEL_INFER" folder, quantizing it to int4, and performing inference.

These two folders, "E2B_INT4_MODEL_INFER" and "E2B_ORIGINAL_MODEL_INFER", are the folders we will work in.
The "E2B_ORIGINAL_MODEL_INFER" folder is for original model inference.
The "E2B_INT4_MODEL_INFER" folder is for quantized model inference.

The [1]st thing to do: The "[Original Model]gemma3NE2B" folder inside "E2B_ORIGINAL_MODEL_INFER" contains the gemma3N E2B model.
And the "E4B_ORIGINAL_MODEL_INFER" folder contains python code, although it's for gemma3N E4B model inference.
Based on this code, create the python files exactly the same inside "E2B_ORIGINAL_MODEL_INFER" and check if it runs by modifying it exclusively for E2B.
Since the gemma3N E series models have a special structure, it is recommended to read the markdown document explaining the E4B model's structure inside the "local_gemma_3n" folder within the "E4B_ORIGINAL_MODEL_INFER" folder.

The [2]nd thing to do is the "E2B_INT4_MODEL_INFER" folder. If task [1] is successful, take the python file produced from task [1] exactly as it is, paste it into "E2B_INT4_MODEL_INFER", then quantize the gemma3N E2B model in "E2B_ORIGINAL_MODEL_INFER", copy it into "E2B_INT4_MODEL_INFER", and then slightly modify the python file exclusively for E2B INT4 quantization.

[Caution]
The output must come out normally.
When inferring the original_model, it normally forwards tokens and generates answers on its own very logically, contextually, and in Korean.
The quantized model must also output in Korean, making sense and fitting the context in the exact same way.
