# Master Directory Guide (Python Software & Controller)

## 1. Execution Environment
* **Virtual Environment:** pynq_env
* **Activation Command:** `source /home/hwkim/Desktop/github/TinyNPU-RTL/pynq_env/bin/activate`
* **Execution Command:** `/home/hwkim/Desktop/github/TinyNPU-RTL/pynq_env/bin/python`

## 2. Current Status & Goals
* **[Current Goal]:** Successfully run the official `google/gemma-3n-E2B-it` model in a local environment in a perfect chat streaming format.
'[Original Model]gemma3NE2B' and '[Original Model]gemma3NE2B_INT4_Q' are the official models.
Inside these folders, there are files such as config.json and tokenizer.json to understand the model's structure.
The conversation must continue naturally like actual commercial models such as GPT or Gemini, not just at a simple output level.
If you need materials, actively utilize official documents or internet searches to verify and confirm. Make sure to memo in debugging.txt!

* **[Pending]:** Complete and verify Python code that quantizes the `gemma3N E2B abilterated` model to INT4 and infers it.
'[abliterated Model]gemma3NE2B_INT4_Q' and '[abliterated Model]gemma3NE2B' are the pending models.

## 3. Strict Rules: debugging.txt
* You must unconditionally record (memo) in the `debugging.txt` file whenever modifying code, when an error occurs, or whenever you acquire knowledge such as model-specific details, model structure, etc.
* **Record Format:** [Attempted Action] - [Occurred Problem/Error Log] - [Solution and Result]
* To avoid repeating the same mistakes, check `debugging.txt` like a notepad before coding to maintain context.
