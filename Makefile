# FaceCV Deployment Makefile

.PHONY: help deploy deploy-quick test-local clean setup-ssh

# Default target
help:
	@echo "FaceCV Deployment Commands:"
	@echo "  make deploy       - Full deployment with confirmation"
	@echo "  make deploy-quick - Quick deployment without restart"
	@echo "  make test-local   - Test locally before deployment"
	@echo "  make setup-ssh    - Setup SSH key for passwordless deployment"
	@echo "  make clean        - Clean local cache and temp files"

# Full deployment with service restart
deploy:
	@echo "🚀 Starting full deployment..."
	@bash scripts/deploy_to_client.sh

# Quick deployment without service restart
deploy-quick:
	@echo "📦 Quick deployment (no service restart)..."
	@sshpass -p 'tdit@2155' rsync -avz --delete \
		--exclude-from='.deployignore' \
		--progress \
		-e "ssh -p 22 -o StrictHostKeyChecking=no" \
		./ tdit@113.44.157.91:/home/tdit/facecv/
	@echo "✅ Files synced! Service not restarted."

# Test locally before deployment
test-local:
	@echo "🧪 Running local tests..."
	@python -m pytest tests/ -v --tb=short || true
	@echo ""
	@echo "🔍 Checking code quality..."
	@python -m flake8 facecv/ --max-line-length=100 --exclude=__pycache__ || true
	@echo ""
	@echo "📋 Checking deployment ignore list..."
	@cat .deployignore

# Setup SSH key for passwordless deployment
setup-ssh:
	@echo "🔑 Setting up SSH key for passwordless deployment..."
	@echo "1. Generate SSH key if not exists:"
	@echo "   ssh-keygen -t rsa -b 4096 -f ~/.ssh/facecv_deploy_key"
	@echo ""
	@echo "2. Copy key to server:"
	@echo "   sshpass -p 'tdit@2155' ssh-copy-id -i ~/.ssh/facecv_deploy_key.pub -p 22 tdit@113.44.157.91"
	@echo ""
	@echo "3. Test connection:"
	@echo "   ssh -i ~/.ssh/facecv_deploy_key -p 22 tdit@113.44.157.91 'echo Connected successfully!'"

# Clean local files
clean:
	@echo "🧹 Cleaning local files..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type f -name "*.log" -delete
	@find . -type f -name "*.db" -delete 2>/dev/null || true
	@echo "✅ Cleanup complete!"

# Variables for deployment
REMOTE_USER = tdit
REMOTE_HOST = 113.44.157.91
REMOTE_PORT = 22
REMOTE_PATH = /home/tdit/facecv