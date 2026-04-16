#!/bin/bash
# ============================================================
# AWS EC2 Deployment — Signature Verification App
# ============================================================
# Deploys directly to an EC2 instance (no Docker needed).
# The server installs Python, clones your repo, and runs the app.
#
# Prerequisites:
#   1. AWS CLI installed and configured (aws configure)
#   2. Your repo pushed to GitHub (with the model file)
#
# Usage:
#   chmod +x deploy/deploy-ec2.sh
#   ./deploy/deploy-ec2.sh
# ============================================================

set -euo pipefail

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
AWS_REGION="${AWS_REGION:-ap-south-1}"
APP_NAME="sig-verify"
KEY_NAME="${APP_NAME}-key"
SG_NAME="${APP_NAME}-sg"
INSTANCE_TYPE="t2.micro"   # Free tier eligible!

# ⚠️ CHANGE THIS to your GitHub repo URL
GITHUB_REPO="https://github.com/suineg-1110/signature_verification.git"

echo ""
echo "============================================"
echo "  AWS EC2 Deployment — Signature Verification"
echo "============================================"
echo ""

# ──────────────────────────────────────────────
# STEP 1: Detect AWS Account
# ──────────────────────────────────────────────
echo "🔍 Step 1: Detecting AWS account..."
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "  Account: ${ACCOUNT_ID}"
echo "  Region:  ${AWS_REGION}"
echo ""

# ──────────────────────────────────────────────
# STEP 2: Create Key Pair (for SSH access)
# ──────────────────────────────────────────────
echo "🔑 Step 2: Creating SSH key pair..."
KEY_FILE="${KEY_NAME}.pem"

if [ -f "${KEY_FILE}" ]; then
    echo "  Key file already exists, skipping..."
else
    # Delete old key if it exists in AWS
    aws ec2 delete-key-pair --key-name "${KEY_NAME}" --region "${AWS_REGION}" 2>/dev/null || true
    
    aws ec2 create-key-pair \
        --key-name "${KEY_NAME}" \
        --query "KeyMaterial" \
        --output text \
        --region "${AWS_REGION}" > "${KEY_FILE}"
    
    chmod 400 "${KEY_FILE}"
    echo "  Created: ${KEY_FILE}"
fi
echo ""

# ──────────────────────────────────────────────
# STEP 3: Get Default VPC
# ──────────────────────────────────────────────
echo "🌐 Step 3: Finding default VPC..."
VPC_ID=$(aws ec2 describe-vpcs \
    --filters "Name=isDefault,Values=true" \
    --query "Vpcs[0].VpcId" \
    --output text \
    --region "${AWS_REGION}")
echo "  VPC: ${VPC_ID}"
echo ""

# ──────────────────────────────────────────────
# STEP 4: Create Security Group
# ──────────────────────────────────────────────
echo "🔒 Step 4: Creating security group..."
SG_ID=$(aws ec2 create-security-group \
    --group-name "${SG_NAME}" \
    --description "Signature Verification App - ports 22, 7860" \
    --vpc-id "${VPC_ID}" \
    --query "GroupId" \
    --output text \
    --region "${AWS_REGION}" 2>/dev/null) || \
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=${SG_NAME}" "Name=vpc-id,Values=${VPC_ID}" \
    --query "SecurityGroups[0].GroupId" \
    --output text \
    --region "${AWS_REGION}")

# Allow SSH (port 22)
aws ec2 authorize-security-group-ingress \
    --group-id "${SG_ID}" \
    --protocol tcp --port 22 --cidr 0.0.0.0/0 \
    --region "${AWS_REGION}" 2>/dev/null || true

# Allow Gradio (port 7860)
aws ec2 authorize-security-group-ingress \
    --group-id "${SG_ID}" \
    --protocol tcp --port 7860 --cidr 0.0.0.0/0 \
    --region "${AWS_REGION}" 2>/dev/null || true

echo "  Security Group: ${SG_ID}"
echo ""

# ──────────────────────────────────────────────
# STEP 5: Find Amazon Linux 2023 AMI
# ──────────────────────────────────────────────
echo "🖥️  Step 5: Finding latest Amazon Linux 2023 AMI..."
AMI_ID=$(aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=al2023-ami-2023*-x86_64" "Name=state,Values=available" \
    --query "Images | sort_by(@, &CreationDate) | [-1].ImageId" \
    --output text \
    --region "${AWS_REGION}")
echo "  AMI: ${AMI_ID}"
echo ""

# ──────────────────────────────────────────────
# STEP 6: Create User Data Script
# ──────────────────────────────────────────────
# This runs automatically when the instance starts.
# It installs Python, clones your repo, and starts the app.

USER_DATA=$(cat << 'USERDATA_EOF'
#!/bin/bash
exec > /var/log/app-setup.log 2>&1
set -ex

echo "========== Starting setup =========="

# Install Python 3.10 and git
dnf update -y
dnf install -y python3.11 python3.11-pip git

# Create app directory
mkdir -p /home/ec2-user/app
cd /home/ec2-user/app

# Clone the repo
git clone GITHUB_REPO_PLACEHOLDER .

# Install CPU-only PyTorch + other dependencies
pip3.11 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip3.11 install -r requirements.txt

# Create a systemd service so the app starts on boot
cat > /etc/systemd/system/sigverify.service << 'SERVICE_EOF'
[Unit]
Description=Signature Verification Gradio App
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/app
ExecStart=/usr/bin/python3.11 gradio_app/app.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
SERVICE_EOF

# Fix ownership
chown -R ec2-user:ec2-user /home/ec2-user/app

# Start the service
systemctl daemon-reload
systemctl enable sigverify
systemctl start sigverify

echo "========== Setup complete! =========="
USERDATA_EOF
)

# Replace placeholder with actual repo URL
USER_DATA="${USER_DATA//GITHUB_REPO_PLACEHOLDER/${GITHUB_REPO}}"

# Base64 encode for AWS
USER_DATA_B64=$(echo "${USER_DATA}" | base64 -w 0)

# ──────────────────────────────────────────────
# STEP 7: Launch EC2 Instance
# ──────────────────────────────────────────────
echo "🚀 Step 6: Launching EC2 instance (${INSTANCE_TYPE})..."

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "${AMI_ID}" \
    --instance-type "${INSTANCE_TYPE}" \
    --key-name "${KEY_NAME}" \
    --security-group-ids "${SG_ID}" \
    --user-data "${USER_DATA_B64}" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${APP_NAME}}]" \
    --query "Instances[0].InstanceId" \
    --output text \
    --region "${AWS_REGION}")

echo "  Instance ID: ${INSTANCE_ID}"
echo ""

# ──────────────────────────────────────────────
# STEP 8: Wait for instance to be running
# ──────────────────────────────────────────────
echo "⏳ Step 7: Waiting for instance to start..."
aws ec2 wait instance-running \
    --instance-ids "${INSTANCE_ID}" \
    --region "${AWS_REGION}"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "${INSTANCE_ID}" \
    --query "Reservations[0].Instances[0].PublicIpAddress" \
    --output text \
    --region "${AWS_REGION}")

echo ""
echo "============================================"
echo "  ✅ EC2 INSTANCE LAUNCHED!"
echo "============================================"
echo ""
echo "  Instance ID: ${INSTANCE_ID}"
echo "  Public IP:   ${PUBLIC_IP}"
echo ""
echo "  📋 The server is now installing Python &"
echo "     dependencies. This takes ~3-5 minutes."
echo ""
echo "  🌐 Your app will be available at:"
echo "     http://${PUBLIC_IP}:7860"
echo ""
echo "  🔑 To SSH into the server:"
echo "     ssh -i ${KEY_FILE} ec2-user@${PUBLIC_IP}"
echo ""
echo "  📋 To check setup progress (SSH in first):"
echo "     tail -f /var/log/app-setup.log"
echo ""
echo "  🛑 To stop (and stop paying):"
echo "     aws ec2 stop-instances --instance-ids ${INSTANCE_ID} --region ${AWS_REGION}"
echo ""
echo "  ▶️  To start again:"
echo "     aws ec2 start-instances --instance-ids ${INSTANCE_ID} --region ${AWS_REGION}"
echo ""
echo "  🗑️  To delete permanently:"
echo "     aws ec2 terminate-instances --instance-ids ${INSTANCE_ID} --region ${AWS_REGION}"
echo ""
