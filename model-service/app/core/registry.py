"""
Service Discovery and Health Monitoring

Tracks availability and health of all model services in the cluster.
Provides GPU statistics and intelligent routing decisions.
"""

from typing import Dict, List, Optional
import httpx
import asyncio
from datetime import datetime
from pydantic import BaseModel

from app.core.logger import setup_logger

logger = setup_logger("service_registry")


class ServiceHealth(BaseModel):
    """Health status of a service"""
    name: str
    url: str
    healthy: bool
    last_check: datetime
    response_time_ms: Optional[float] = None
    error: Optional[str] = None


class GPUStats(BaseModel):
    """GPU usage statistics from a service"""
    device_id: int
    name: str
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    temperature_c: Optional[float] = None
    utilization_percent: Optional[float] = None


class ServiceRegistry:
    """
    Central registry for tracking all model services in the cluster.
    
    Provides health checks, GPU monitoring, and service discovery.
    """
    
    def __init__(self):
        """Initialize service registry with known endpoints"""
        self.services = {
            "llm-primary": "http://llm-primary:8000",
            "llm-secondary": "http://llm-secondary:8001",
            "embedding": "http://embedding:8002",
            "multimodal": "http://multimodal:8003",
            # "transcription": "http://transcription:8004",  # Optional
        }
        self._health_cache: Dict[str, ServiceHealth] = {}
        self._gpu_cache: Dict[str, List[GPUStats]] = {}
        self._cache_duration_seconds = 5  # Cache results for 5 seconds
    
    async def health_check(self, force_refresh: bool = False) -> Dict[str, ServiceHealth]:
        """
        Check health of all services
        
        Args:
            force_refresh: Force fresh health check, ignoring cache
        
        Returns:
            Dictionary mapping service names to health status
        """
        # Return cached results if recent enough
        if not force_refresh and self._health_cache:
            first_check = list(self._health_cache.values())[0]
            age = (datetime.now() - first_check.last_check).total_seconds()
            if age < self._cache_duration_seconds:
                return self._health_cache
        
        results = {}
        
        async with httpx.AsyncClient() as client:
            tasks = []
            for name, url in self.services.items():
                tasks.append(self._check_single_service(client, name, url))
            
            health_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for health in health_results:
                if isinstance(health, ServiceHealth):
                    results[health.name] = health
                elif isinstance(health, Exception):
                    logger.error(f"Health check failed: {health}")
        
        self._health_cache = results
        return results
    
    async def _check_single_service(
        self, 
        client: httpx.AsyncClient, 
        name: str, 
        url: str
    ) -> ServiceHealth:
        """Check health of a single service"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            response = await client.get(
                f"{url}/health",
                timeout=2.0
            )
            
            elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return ServiceHealth(
                name=name,
                url=url,
                healthy=response.status_code == 200,
                last_check=datetime.now(),
                response_time_ms=round(elapsed_ms, 2)
            )
        
        except Exception as e:
            return ServiceHealth(
                name=name,
                url=url,
                healthy=False,
                last_check=datetime.now(),
                error=str(e)
            )
    
    async def get_gpu_stats(self, force_refresh: bool = False) -> Dict[str, List[GPUStats]]:
        """
        Get GPU statistics from all services
        
        Args:
            force_refresh: Force fresh GPU stats, ignoring cache
        
        Returns:
            Dictionary mapping service names to GPU statistics
        """
        # Return cached results if recent enough
        if not force_refresh and self._gpu_cache:
            return self._gpu_cache
        
        stats = {}
        
        async with httpx.AsyncClient() as client:
            for name, url in self.services.items():
                try:
                    response = await client.get(
                        f"{url}/gpu-stats",
                        timeout=2.0
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if "gpus" in data:
                            stats[name] = [GPUStats(**gpu) for gpu in data["gpus"]]
                    else:
                        stats[name] = []
                
                except Exception as e:
                    logger.debug(f"Failed to get GPU stats from {name}: {e}")
                    stats[name] = []
        
        self._gpu_cache = stats
        return stats
    
    async def get_available_services(self, service_type: str = "llm") -> List[str]:
        """
        Get list of healthy services of a specific type
        
        Args:
            service_type: Type of service to filter for ("llm", "embedding", etc.)
        
        Returns:
            List of service names that are healthy
        """
        health = await self.health_check()
        
        available = []
        for name, status in health.items():
            if status.healthy and service_type in name:
                available.append(name)
        
        return available
    
    async def get_best_service(self, service_type: str = "llm") -> Optional[str]:
        """
        Get the best available service based on health and response time
        
        Args:
            service_type: Type of service needed
        
        Returns:
            Service name or None if no healthy services available
        """
        available = await self.get_available_services(service_type)
        
        if not available:
            return None
        
        # Sort by response time (fastest first)
        health = await self.health_check()
        sorted_services = sorted(
            available,
            key=lambda x: health[x].response_time_ms or float('inf')
        )
        
        return sorted_services[0] if sorted_services else None
    
    def get_service_url(self, service_name: str) -> Optional[str]:
        """Get URL for a specific service"""
        return self.services.get(service_name)


# Global registry instance
registry = ServiceRegistry()
