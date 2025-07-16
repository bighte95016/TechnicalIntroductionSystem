# cache_augmented_generation.py

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch


def load_transformer_model(model_id="circulus/gemma-3-12b-it-gptq"):
    print(f"[CAG] ⬇️  Loading Gemma3 4-bit model from: {model_id}")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=torch.float16,
    )

    return tokenizer, model


def generate_response(prompt, tokenizer, model, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def build_cache():
    tokenizer, model = load_transformer_model("google/gemma-2b-it")
    prompt = "Q: What is Cache Augmented Generation?\nA:"
    answer = generate_response(prompt, tokenizer, model)
    print("[CAG] 🤖 回答：", answer)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cache-Augmented Generation with Gemma3")
    parser.add_argument("--build_cache", action="store_true", help="Build the cache (run generation)")
    args = parser.parse_args()

    if args.build_cache:
        build_cache()
