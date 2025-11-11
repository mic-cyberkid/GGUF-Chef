# chat_llm_service.py
import os
import logging
from chat_utils import SYSTEM_PROMPT
from llama_cpp import Llama
from tech_support_logger import TechSupportLogger

llm_logger = TechSupportLogger(
    log_file_name="llm_service.log",
    log_dir="data/logs",
    level=logging.INFO,
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_output=False
).get_logger()

MODEL_DIR = "models/"
MODEL_FILENAME = "PhysicsChatBot.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

GENERATION_PARAMS = {
    "stop": ["<|im_end|>"],
    "max_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
}

class ChatLLMService:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=0,
            use_mmap=True,
            verbose=False,
        )
        llm_logger.info("LLM model loaded from: %s", model_path)

    def format_prompt(self, message, history, _):  # Ignore chunks
        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"

        # Use only last 2 turns from history
        for msg in history[-2:]:
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"

        prompt += f"<|im_start|>user\n{message}\n\n<|im_end|>\n<|im_start|>assistant:\n<think>\n\n</think>\n\n"

        return prompt

    def generate_response(self, prompt: str) -> str:
        try:
            output = self.model(prompt, **GENERATION_PARAMS)
            return output['choices'][0]['text'].strip()
        except Exception as e:
            llm_logger.error("LLM generation error: %s", e, exc_info=True)
            raise RuntimeError("Failed to generate LLM response.") from e

    def generate_response_stream(self, prompt: str):
        """
        Generator that streams LLM output token-by-token.
        """
        def token_stream():
            try:
                for output in self.model(prompt, stream=True, **GENERATION_PARAMS):
                    token = output["choices"][0]["text"]
                    yield token
            except Exception as e:
                llm_logger.error("LLM streaming error: %s", e, exc_info=True)
                yield "[ERROR] Streaming failed.\n"

        return token_stream()
