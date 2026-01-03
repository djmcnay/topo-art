import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

# Load raw config
with open("auth_config_raw.yaml", "r") as file:
    config = yaml.load(file, Loader=SafeLoader)

# Extract plaintext passwords
passwords = []
for user in config['credentials']['usernames'].values():
    passwords.append(user['password'])

# Hash passwords using the correct method for this version
hashed_passwords = stauth.Hasher.hash_list(passwords)

# Update config with hashed passwords
for (user, hash_pw) in zip(config['credentials']['usernames'].values(), hashed_passwords):
    user['password'] = hash_pw

# Save updated config
with open("auth_config.yaml", "w") as file:
    yaml.dump(config, file)

print("✅ Passwords hashed and saved to .streamlit/auth_config.yaml")
