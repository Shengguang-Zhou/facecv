#!/bin/bash
# FaceCV Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="facecv"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-}"
VERSION="${VERSION:-latest}"
ENVIRONMENT="${ENVIRONMENT:-production}"

echo -e "${GREEN}FaceCV Deployment Script${NC}"
echo "================================"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check .env file
    if [ ! -f .env ]; then
        print_warning ".env file not found. Creating from example..."
        cp .env.example .env
    fi
    
    print_status "Prerequisites check completed"
}

# Build Docker image
build_image() {
    print_status "Building Docker image..."
    
    if [ "$ENVIRONMENT" == "development" ]; then
        docker build --target development -t ${PROJECT_NAME}:dev .
    else
        docker build --target production -t ${PROJECT_NAME}:${VERSION} .
        docker tag ${PROJECT_NAME}:${VERSION} ${PROJECT_NAME}:latest
    fi
    
    print_status "Docker image built successfully"
}

# Push to registry (if configured)
push_image() {
    if [ -n "$DOCKER_REGISTRY" ]; then
        print_status "Pushing image to registry..."
        
        docker tag ${PROJECT_NAME}:${VERSION} ${DOCKER_REGISTRY}/${PROJECT_NAME}:${VERSION}
        docker tag ${PROJECT_NAME}:${VERSION} ${DOCKER_REGISTRY}/${PROJECT_NAME}:latest
        
        docker push ${DOCKER_REGISTRY}/${PROJECT_NAME}:${VERSION}
        docker push ${DOCKER_REGISTRY}/${PROJECT_NAME}:latest
        
        print_status "Image pushed to registry"
    fi
}

# Deploy with Docker Compose
deploy_compose() {
    print_status "Deploying with Docker Compose..."
    
    if [ "$ENVIRONMENT" == "development" ]; then
        docker-compose --profile development up -d
    else
        docker-compose up -d
    fi
    
    # Wait for services to be healthy
    print_status "Waiting for services to be healthy..."
    sleep 10
    
    # Check service health
    if docker-compose ps | grep -q "unhealthy"; then
        print_error "Some services are unhealthy"
        docker-compose ps
        exit 1
    fi
    
    print_status "Deployment completed successfully"
}

# Run database migrations
run_migrations() {
    print_status "Running database migrations..."
    
    docker-compose exec -T facecv-api python -c "
from facecv.database.factory import create_face_database
from facecv.config.database import DatabaseConfig

config = DatabaseConfig()
db = create_face_database()
print('Database initialized successfully')
"
    
    print_status "Migrations completed"
}

# Health check
health_check() {
    print_status "Performing health check..."
    
    # Wait for API to be ready
    for i in {1..30}; do
        if curl -f http://localhost:7000/health > /dev/null 2>&1; then
            print_status "API is healthy"
            return 0
        fi
        echo -n "."
        sleep 2
    done
    
    print_error "API health check failed"
    return 1
}

# Show deployment info
show_info() {
    print_status "Deployment Information:"
    echo "================================"
    echo "Environment: $ENVIRONMENT"
    echo "Version: $VERSION"
    echo "API URL: http://localhost:7000"
    echo "API Docs: http://localhost:7000/docs"
    echo "================================"
    
    if [ "$ENVIRONMENT" == "production" ]; then
        echo "Services running:"
        docker-compose ps
    fi
}

# Main deployment flow
main() {
    case "${1:-deploy}" in
        build)
            check_prerequisites
            build_image
            ;;
        push)
            push_image
            ;;
        deploy)
            check_prerequisites
            build_image
            deploy_compose
            run_migrations
            health_check
            show_info
            ;;
        stop)
            print_status "Stopping services..."
            docker-compose down
            ;;
        restart)
            print_status "Restarting services..."
            docker-compose restart
            ;;
        logs)
            docker-compose logs -f ${2:-facecv-api}
            ;;
        status)
            docker-compose ps
            ;;
        *)
            echo "Usage: $0 {build|push|deploy|stop|restart|logs|status}"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"