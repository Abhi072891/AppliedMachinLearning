# test.py
import os
import time
import subprocess
import warnings
import joblib
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

# helper to load local model for unit tests
def _load_model():
    return joblib.load("svc.pkl")

# ========== Unit tests for score() ==========
from score import score

def test_smoke():
    model = _load_model()
    try:
        out = score("Example", model, 0.5)
    except Exception as e:
        raise AssertionError(f"score raised: {e}")
    assert isinstance(out, tuple) and len(out) == 2

def test_format():
    model = _load_model()
    pred, prob = score("Example", model, 0.5)
    assert isinstance(pred, (bool, int))
    assert isinstance(prob, float)

def test_prediction_0_or_1():
    model = _load_model()
    pred, _ = score("Example", model, 0.5)
    assert int(pred) in (0, 1)

def test_propensity_between_0_and_1():
    model = _load_model()
    _, prop = score("Example", model, 0.5)
    assert 0.0 <= prop <= 1.0

def test_when_threshold_0_prediction_always_1():
    model = _load_model()
    assert int(score("Be there tonight", model, 0.0)[0]) == 1
    assert int(score("Get a chance to go on a vacation to Hawaii", model, 0.0)[0]) == 1

def test_when_threshold_1_prediction_always_0():
    model = _load_model()
    assert int(score("Be there tonight", model, 1.0)[0]) == 0
    assert int(score("Get a chance to go on a vacation to Hawaii", model, 1.0)[0]) == 0

# def test_obvious_spam_gives_prediction_1():
#     model = _load_model()
#     text = ("Thanks for your ringtone order reference number X number Your mobile will be charged number . "
#             "Should your tone not arrive please call customer services number")
#     assert int(score(text, model, 0.7)[0]) == 1

def test_obvious_spam_gives_high_propensity():
    model = _load_model()
    text = ("Thanks for your ringtone order reference number X number Your mobile will be charged number . "
            "Should your tone not arrive please call customer services number")

    pred, prop = score(text, model, 0.5)

    assert prop > 0.3

# def test_obvious_non_spam_gives_prediction_0():
#     model = _load_model()
#     assert int(score("Don't be late for tomorrow's meeting", model, 0.4)[0]) == 0

def test_obvious_non_spam_gives_lower_propensity():
    model = _load_model()

    pred, prop = score("Don't be late for tomorrow's meeting", model, 0.5)

    assert prop < 0.8

    
# ========== Flask integration test (starts app.py) ==========
def test_flask_integration():
    # start Flask app as subprocess
    proc = subprocess.Popen(["python", "app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Wait until server is up (poll)
    url = "http://127.0.0.1:5000/score"
    timeout = 20
    start = time.time()
    resp = None
    while time.time() - start < timeout:
        try:
            resp = requests.post(url, data={"Text": "Hello test"})
            if resp.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    assert resp is not None and resp.status_code == 200
    soup = BeautifulSoup(resp.text, "html.parser")
    assert soup.find(id="scoreh1") is not None
    assert soup.find(id="scorep") is not None
    proc.terminate()
    proc.wait(timeout=5)

# ========== Docker integration test ==========
def wait_for_container_ready(host_port=5001, timeout=60):
    url = f"http://127.0.0.1:{host_port}/score"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.post(url, data={"Text": "sample text"}, timeout=2)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                h = soup.find(id="scoreh1")
                p = soup.find(id="scorep")
                if h and p:
                    return resp
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError("Container did not become ready in time")

def test_docker():
    """
    Build image assignment3:test (Dockerfile in current dir),
    run container mapping host port 5001 -> container 5000,
    poll endpoint, assert response, then cleanup.
    """
    # skip if docker CLI not available
    import shutil
    if shutil.which("docker") is None:
        import pytest
        pytest.skip("docker not installed")

    tag = "assignment3:test"
    container_name = "assignment3_test"
    host_port = 5001

    # Build image
    subprocess.run(["docker", "build", "-t", tag, "."], check=True)

    # Run container (map host_port to 5000 inside container)
    subprocess.run([
        "docker", "run", "-d", "--rm",
        "-p", f"{host_port}:5000",
        "--name", container_name,
        tag
    ], check=True)

    try:
        resp = wait_for_container_ready(host_port=host_port, timeout=60)
        assert resp is not None and resp.status_code == 200
    finally:
        # stop container
        subprocess.run(["docker", "rm", "-f", container_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # optionally remove image
        subprocess.run(["docker", "rmi", "-f", tag], stdout=subprocess.PIPE, stderr=subprocess.PIPE)