from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import requests
from requests.exceptions import RequestException


def asr():
    url = "http://192.168.100.111:8000/v1/audio/transcriptions"

    filename = "sample_10s.mp3"
    payload = dict(
        file=open(filename, "rb"),
        model=(None, "ichigo"),
    )

    t0 = time.perf_counter()
    resp = requests.post(url, files=payload)
    resp.raise_for_status()
    t1 = time.perf_counter()

    return t1 - t0


def main():
    num_concurrent_calls = 20

    print(f"Running {num_concurrent_calls} concurrent API calls...")
    with ThreadPoolExecutor(max_workers=num_concurrent_calls) as executor:
        futures = [executor.submit(asr) for _ in range(num_concurrent_calls)]

        for i, future in enumerate(as_completed(futures)):
            latency = future.result()
            print(f"Call {i}: Latency = {latency:.3f} seconds")


if __name__ == "__main__":
    main()
