"""Generate controlled endpoint traffic and a lightweight latency summary."""
from time import perf_counter
from statistics import mean


def generate_monitoring_requests(ml_client, endpoint_name, deployment_name,
                                 request_file, request_count=10,
                                 allow_requests=False):
    if not allow_requests:
        return {"status": "SKIPPED", "reason": "Requests are disabled."}
    successes = 0
    for _ in range(request_count):
        response = ml_client.online_endpoints.invoke(
            endpoint_name=endpoint_name,
            deployment_name=deployment_name,
            request_file=str(request_file),
        )
        successes += int(response is not None and str(response).strip() != "")
    return {"status": "COMPLETED", "requests": request_count,
            "successful_responses": successes}


def run_lightweight_benchmark(ml_client, endpoint_name, deployment_name,
                              request_file, request_count=5,
                              allow_benchmark=False):
    if not allow_benchmark:
        return {"status": "SKIPPED", "reason": "Benchmark is disabled."}
    latencies = []
    for _ in range(request_count):
        started = perf_counter()
        ml_client.online_endpoints.invoke(
            endpoint_name=endpoint_name,
            deployment_name=deployment_name,
            request_file=str(request_file),
        )
        latencies.append(perf_counter() - started)
    ordered = sorted(latencies)
    p95_index = max(0, min(len(ordered)-1, round(0.95 * len(ordered)) - 1))
    return {
        "status": "COMPLETED",
        "requests": request_count,
        "mean_seconds": round(mean(latencies), 3),
        "minimum_seconds": round(min(latencies), 3),
        "maximum_seconds": round(max(latencies), 3),
        "p95_seconds": round(ordered[p95_index], 3),
    }


if __name__ == "__main__":
    print("Import this module from Project_2_Master_Notebook_SDK_v2_Working.ipynb.")