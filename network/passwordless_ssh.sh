#!/bin/bash

read -p "Enter the hostname or IP of the remote Mac: " REMOTE_HOST
read -p "Enter the username on the remote Mac: " REMOTE_USER
read -s -p "Enter the password for $REMOTE_USER@$REMOTE_HOST: " PASSWORD
echo ""
read -p "Enter the SSH key name (e.g., id_rsa_custom): " KEY_NAME

SSH_KEY="$HOME/.ssh/$KEY_NAME"

# 1️⃣ Check if SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    echo "No SSH key named $KEY_NAME found. Generating a new SSH key pair..."
    ssh-keygen -t rsa -b 4096 -f "$SSH_KEY" -N ""
else
    echo "SSH key $KEY_NAME already exists."
fi

# 2️⃣ Copy SSH key using sshpass
echo "Copying SSH key to $REMOTE_USER@$REMOTE_HOST..."
sshpass -p "$PASSWORD" ssh-copy-id -o StrictHostKeyChecking=no -i "$SSH_KEY.pub" "$REMOTE_USER@$REMOTE_HOST"

if [ $? -ne 0 ]; then
    echo "❌ Failed to copy SSH key to $REMOTE_USER@$REMOTE_HOST."
    exit 1
fi

# 3️⃣ Test SSH connection
echo "Testing passwordless SSH connection..."
ssh -o BatchMode=yes -o StrictHostKeyChecking=no -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" "echo SSH connection successful."

if [ $? -eq 0 ]; then
    echo "✅ Passwordless SSH setup complete. You can SSH without a password."
else
    echo "❌ Passwordless SSH test failed. You may still be prompted for a password."
    exit 1
fi

# chmod +x setup_passwordless_ssh.sh
