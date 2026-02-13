import json
import os
from cryptography.fernet import Fernet

def generate_rap(config_path="config.json", output_path="rap_data.bin"):
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return

    # 1. Read config.json
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    
    config_bytes = json.dumps(config_data).encode("utf-8")

    # 2. Generate Key
    key = Fernet.generate_key()
    cipher_suite = Fernet(key)

    # 3. Encrypt
    encrypted_data = cipher_suite.encrypt(config_bytes)

    # 4. Save
    with open(output_path, "wb") as f:
        f.write(encrypted_data)

    print(f"RAP data successfully generated and saved to: {output_path}")
    print("\n" + "="*50)
    print("IMPORTANT: Set the following environment variables to use RAP:")
    print(f"RAP_KEY={key.decode()}")
    print("RAP_URL=<URL_TO_YOUR_UPLOADED_RAP_DATA_BIN>")
    print("="*50)

if __name__ == "__main__":
    generate_rap()
