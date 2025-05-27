#!/bin/bash

# Deployment script for FaceCV to client server
# This script syncs code while excluding sensitive and unnecessary files

set -e  # Exit on error

# Configuration
REMOTE_USER="tdit"
REMOTE_HOST="113.44.157.91"
REMOTE_PORT="22"
REMOTE_PATH="/home/tdit/facecv"
LOCAL_PATH="."

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting FaceCV deployment to client server${NC}"
echo -e "Target: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"

# Check if .deployignore exists
if [ ! -f ".deployignore" ]; then
    echo -e "${RED}❌ .deployignore file not found!${NC}"
    exit 1
fi

# Function to check if rsync is installed
check_rsync() {
    if ! command -v rsync &> /dev/null; then
        echo -e "${RED}❌ rsync is not installed. Please install it first.${NC}"
        exit 1
    fi
}

# Function to perform deployment
deploy() {
    echo -e "${YELLOW}📦 Syncing files to remote server...${NC}"
    
    # Use rsync with exclude-from for deployment
    rsync -avz --delete \
        --exclude-from='.deployignore' \
        --progress \
        -e "ssh -p ${REMOTE_PORT}" \
        "${LOCAL_PATH}/" \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Files synced successfully!${NC}"
    else
        echo -e "${RED}❌ Deployment failed!${NC}"
        exit 1
    fi
}

# Function to restart service on remote server
restart_service() {
    echo -e "${YELLOW}🔄 Restarting FaceCV service on remote server...${NC}"
    
    ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} << 'EOF'
        cd /home/tdit/facecv
        
        # Kill existing process if running
        if [ -f facecv.pid ]; then
            PID=$(cat facecv.pid)
            if ps -p $PID > /dev/null 2>&1; then
                echo "Stopping existing FaceCV process (PID: $PID)..."
                kill $PID
                sleep 2
            fi
        fi
        
        # Start the service
        echo "Starting FaceCV service..."
        nohup python main.py > facecv.log 2>&1 &
        echo $! > facecv.pid
        
        # Wait and check if service started
        sleep 3
        if ps -p $(cat facecv.pid) > /dev/null 2>&1; then
            echo "✅ FaceCV service started successfully!"
            echo "📝 Logs: tail -f /home/tdit/facecv/facecv.log"
        else
            echo "❌ Failed to start FaceCV service!"
            tail -n 20 facecv.log
            exit 1
        fi
EOF
}

# Function to check deployment
check_deployment() {
    echo -e "${YELLOW}🔍 Checking deployment status...${NC}"
    
    ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} << 'EOF'
        cd /home/tdit/facecv
        echo "Current directory: $(pwd)"
        echo "Python version: $(python --version)"
        echo ""
        echo "Recent files updated:"
        find . -type f -name "*.py" -mmin -5 | head -10
        echo ""
        if [ -f facecv.pid ] && ps -p $(cat facecv.pid) > /dev/null 2>&1; then
            echo "✅ FaceCV service is running (PID: $(cat facecv.pid))"
        else
            echo "❌ FaceCV service is not running"
        fi
EOF
}

# Main deployment flow
main() {
    check_rsync
    
    # Ask for confirmation
    echo -e "${YELLOW}⚠️  This will sync code to ${REMOTE_HOST}${NC}"
    echo -e "Excluded items:"
    echo "- Environment files (.env*)"
    echo "- Database files (*.db)"
    echo "- Log files (*.log)"
    echo "- Virtual environment (.venv/)"
    echo "- Test files (tests/)"
    echo "- Model weights (weights/)"
    echo "- Documentation (docs/)"
    echo "- Examples (examples/)"
    echo "- Data directories (data/, chroma*/)"
    echo ""
    read -p "Continue with deployment? (y/N) " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Deployment cancelled.${NC}"
        exit 1
    fi
    
    # Perform deployment
    deploy
    
    # Ask if service should be restarted
    echo ""
    read -p "Restart FaceCV service on remote server? (y/N) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        restart_service
    fi
    
    # Check deployment status
    check_deployment
    
    echo -e "${GREEN}🎉 Deployment completed!${NC}"
}

# Run main function
main