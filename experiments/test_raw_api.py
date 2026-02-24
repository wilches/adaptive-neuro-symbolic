"""Minimal test — verify SiliconFlow API key works."""

import sys
import os

def test_raw():
    try:
        from openai import OpenAI
    except ImportError:
        print("Instalando openai...")
        os.system("pip install openai")
        from openai import OpenAI

    # ── Obtener la key ───────────────────────────
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.getenv("SILICONFLOW_API_KEY", "")

    if not api_key:
        api_key = input("Pega tu SiliconFlow API key aqui: ").strip()

    if not api_key:
        print("ERROR: No API key provided")
        sys.exit(1)

    print(f"\nAPI Key: {api_key[:8]}...{api_key[-4:]}")
    print(f"Length:  {len(api_key)} characters")

    # ══════════════════════════════════════════════
    # IMPORTANTE: Usamos la libreria 'openai' como
    # HERRAMIENTA DE CONEXION. No estamos llamando
    # a OpenAI. Estamos llamando a SiliconFlow.
    # La libreria openai es solo el "conector".
    # ══════════════════════════════════════════════

    print("\nConnecting to SiliconFlow...")
    print("(Using openai library as connector — NOT calling OpenAI servers)")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.siliconflow.com/v1",  # ← SiliconFlow, NO OpenAI
    )

    # ── Test 1: Listar modelos disponibles ───────
    print("\n--- Test 1: Listing available models ---")
    try:
        models = client.models.list()
        deepseek_models = [m.id for m in models.data if 'deepseek' in m.id.lower()]
        qwen_models = [m.id for m in models.data if 'qwen' in m.id.lower()]

        print(f"Total models available: {len(models.data)}")
        print(f"\nDeepSeek models:")
        for m in deepseek_models[:10]:
            print(f"  - {m}")
        print(f"\nQwen models:")
        for m in qwen_models[:10]:
            print(f"  - {m}")

    except Exception as e:
        print(f"Error: {e}")
        print("\nPossible fixes:")
        print("  1. Check if API key is correct")
        print("  2. Check internet connection")
        print("  3. Try: https://api.siliconflow.cn/v1 instead")
        return False

    # ── Test 2: Chat simple ──────────────────────
    print("\n--- Test 2: Simple math question ---")
    try:
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            messages=[
                {"role": "user", "content": "What is 2 + 2? Reply with just the number."}
            ],
            max_tokens=100,
            temperature=0,
        )

        answer = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else "unknown"

        print(f"Response: {answer}")
        print(f"Tokens:   {tokens}")
        print("\n✅ SiliconFlow API is working!")
        return True

    except Exception as e:
        print(f"Chat error: {e}")

        # Si el modelo no existe, intentar con otro
        print("\nTrying alternative model: Qwen/Qwen2.5-7B-Instruct...")
        try:
            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=[
                    {"role": "user", "content": "What is 2 + 2? Reply with just the number."}
                ],
                max_tokens=100,
                temperature=0,
            )
            answer = response.choices[0].message.content
            print(f"Response: {answer}")
            print("\n✅ API works! But use model: Qwen/Qwen2.5-7B-Instruct")
            print("   Update src/config.py with this model name.")
            return True
        except Exception as e2:
            print(f"Also failed: {e2}")
            return False


if __name__ == "__main__":
    success = test_raw()
    if success:
        print("\n" + "=" * 50)
        print("NEXT STEP: Run the full API connection test:")
        print("  python experiments/test_api_connection.py")
    else:
        print("\n" + "=" * 50)
        print("TROUBLESHOOTING:")
        print("  1. Go to https://cloud.siliconflow.cn/account/ak")
        print("  2. Create a new API key")
        print("  3. Put it in .env file: SILICONFLOW_API_KEY=sk-...")
        print("  4. Run this script again")
