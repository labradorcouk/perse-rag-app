@echo off
echo 🚀 Starting optimized deployment for Windows...

REM 1. Stop existing containers
echo 🛑 Stopping existing containers...
docker-compose down

REM 2. Clean up old images and containers
echo 🧹 Cleaning up old images and containers...
docker system prune -f
docker volume prune -f

REM 3. Build optimized image
echo 🔨 Building optimized Docker image...
docker-compose -f docker-compose.optimized.windows.yml build --no-cache

REM 4. Deploy with optimized configuration
echo 🚀 Deploying with optimized configuration...
docker-compose -f docker-compose.optimized.windows.yml up -d

REM 5. Wait for services to be healthy
echo ⏳ Waiting for services to be healthy...
timeout /t 30 /nobreak > nul

REM 6. Check service health
echo 🏥 Checking service health...
docker-compose -f docker-compose.optimized.windows.yml ps

REM 7. Monitor performance
echo 📊 Monitoring performance...
echo Docker stats:
docker stats --no-stream

echo ✅ Optimized deployment completed!
echo.
echo 📋 Deployment summary:
echo - Built optimized Docker image
echo - Deployed with resource limits
echo - Configured persistent storage
echo - Set up health monitoring
echo.
echo 🔍 Monitoring commands:
echo - Container status: docker-compose -f docker-compose.optimized.windows.yml ps
echo - Container logs: docker-compose -f docker-compose.optimized.windows.yml logs
echo - Performance stats: docker stats
echo.
echo 🌐 Application URL: http://localhost:8501
pause 