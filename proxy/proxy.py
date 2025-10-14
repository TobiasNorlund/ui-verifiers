from fastapi import FastAPI, Request, Response
from fastapi.responses import Response as FastAPIResponse
from kubernetes import client, config
import httpx

app = FastAPI()

# Load Kubernetes configuration (in-cluster or out-of-cluster)
config.load_incluster_config()
k8s_api = client.CoreV1Api()

# Create a shared HTTP client for better performance
http_client = httpx.AsyncClient(
    timeout=30.0,  # 30 second timeout
    limits=httpx.Limits(
        max_keepalive_connections=20,
        max_connections=100
    )
)

@app.api_route('/proxy/{session_id}/{endpoint:path}', methods=['GET', 'POST', 'PUT', 'DELETE'])
async def proxy_request(session_id: str, endpoint: str, request: Request):
    try:
        # Find the pod using a label selector
        pod_list = k8s_api.list_namespaced_pod(
            namespace='default',
            label_selector=f'session-id={session_id}'
        )

        if not pod_list.items:
            return FastAPIResponse(content="Pod not found", status_code=404)

        pod_ip = pod_list.items[0].status.pod_ip
        pod_port = 8000 # Assuming your API server is on port 8000

        # Forward the request to the pod
        pod_url = f"http://{pod_ip}:{pod_port}/{endpoint}"
        
        # Preserve query parameters if they exist
        if request.query_params:
            pod_url += f"?{request.query_params}"

        # Get request body
        body = await request.body()
        
        # Use the shared HTTP client for better performance
        resp = await http_client.request(
            method=request.method,
            url=pod_url,
            headers={key: value for (key, value) in request.headers.items() if key != 'host'},
            content=body,
            follow_redirects=False
        )
        return FastAPIResponse(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers)
        )

    except Exception as e:
        return FastAPIResponse(content=str(e), status_code=500)

# Cleanup function to close the HTTP client
@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)