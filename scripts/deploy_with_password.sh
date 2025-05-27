#!/bin/bash

# Deployment script using password authentication
# Requires sshpass to be installed: sudo apt-get install sshpass

set -e  # Exit on error

# Configuration
REMOTE_USER="tdit"
REMOTE_HOST="113.44.157.91"
REMOTE_PORT="22"
REMOTE_PATH="/home/tdit/facecv"
REMOTE_PASS="tdit@2155"
LOCAL_PATH="."

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting FaceCV deployment to client server${NC}"
echo -e "Target: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"

# Check if sshpass is installed
if ! command -v sshpass &> /dev/null; then
    echo -e "${RED}❌ sshpass is not installed.${NC}"
    echo "Install it with: sudo apt-get install sshpass"
    exit 1
fi

# Check if rsync is installed
if ! command -v rsync &> /dev/null; then
    echo -e "${RED}❌ rsync is not installed.${NC}"
    echo "Install it with: sudo apt-get install rsync"
    exit 1
fi

# Deploy function
deploy() {
    echo -e "${YELLOW}📦 Syncing files to remote server...${NC}"
    
    # Use rsync with sshpass for password authentication
    sshpass -p "${REMOTE_PASS}" rsync -avz --delete \
        --exclude-from='.deployignore' \
        --progress \
        -e "ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no" \
        "${LOCAL_PATH}/" \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Files synced successfully!${NC}"
    else
        echo -e "${RED}❌ Deployment failed!${NC}"
        exit 1
    fi
}

# Main execution
echo -e "${YELLOW}⚠️  Starting deployment...${NC}"
deploy

echo -e "${GREEN}🎉 Deployment completed!${NC}"
echo ""
echo "To restart the service on the remote server, run:"
echo "ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST}"
echo "Then: cd ${REMOTE_PATH} && python main.py"