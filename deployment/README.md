# FaceCV Deployment Guide

This guide explains how to deploy FaceCV to your client's server using various methods.

## Prerequisites

- `rsync` installed on your local machine
- `sshpass` for password-based deployment (optional)
- SSH access to the target server

## Deployment Methods

### 1. Quick Deployment (Recommended)

```bash
make deploy-quick
```

This syncs all files except those listed in `.deployignore` without restarting the service.

### 2. Full Deployment with Service Restart

```bash
make deploy
```

This will:
- Sync all files
- Ask for confirmation
- Optionally restart the FaceCV service
- Verify deployment status

### 3. Manual Deployment Script

```bash
./scripts/deploy_with_password.sh
```

Uses password authentication for simple deployment.

### 4. GitHub Actions (CI/CD)

Push to `master` branch to trigger automatic deployment.

**Setup required:**
1. Add SSH key to GitHub Secrets as `DEPLOY_SSH_KEY`
2. Configure server to accept the SSH key

## Excluded Files

The following are NOT deployed (see `.deployignore`):

- `.env`, `.env.*` - Environment configuration files
- `*.db` - Database files
- `*.log` - Log files
- `.venv/`, `venv/` - Virtual environment
- `tests/` - Test files
- `weights/` - Model weight files
- `docs/` - Documentation
- `examples/` - Example code
- `data/`, `chroma*/` - Data directories

## Server Configuration

**Target Server:**
- Host: 113.44.157.91
- User: tdit
- Path: /home/tdit/facecv
- Port: 22

## Setting Up Passwordless SSH (Optional)

For more secure and convenient deployment:

```bash
# Generate SSH key
ssh-keygen -t rsa -b 4096 -f ~/.ssh/facecv_deploy_key

# Copy to server (requires password once)
sshpass -p 'tdit@2155' ssh-copy-id -i ~/.ssh/facecv_deploy_key.pub -p 22 tdit@113.44.157.91

# Test connection
ssh -i ~/.ssh/facecv_deploy_key -p 22 tdit@113.44.157.91 'echo Connected!'
```

Then update deployment scripts to use the key instead of password.

## Deployment Workflow

1. **Before Deployment:**
   ```bash
   # Test locally
   make test-local
   
   # Clean unnecessary files
   make clean
   ```

2. **Deploy:**
   ```bash
   # Quick sync
   make deploy-quick
   
   # OR full deployment
   make deploy
   ```

3. **After Deployment:**
   - Check server logs: `ssh tdit@113.44.157.91 'tail -f /home/tdit/facecv/facecv.log'`
   - Verify service: `ssh tdit@113.44.157.91 'ps aux | grep python'`

## Troubleshooting

### Permission Denied
- Ensure password is correct
- Check if SSH service is running on server
- Verify port 22 is open

### Files Not Syncing
- Check `.deployignore` file
- Ensure rsync is installed: `sudo apt-get install rsync`

### Service Not Starting
- Check Python environment on server
- Verify all dependencies are installed
- Check log files for errors

## Security Notes

1. **Never commit `.env` files** to version control
2. **Use SSH keys** instead of passwords when possible
3. **Regularly update** dependencies on the server
4. **Monitor logs** for any security issues