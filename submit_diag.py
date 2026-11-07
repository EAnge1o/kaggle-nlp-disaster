"""Diagnose + execute the competition submission with per-step timing and timeouts."""
import os
import time

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import (
    ApiStartSubmissionUploadRequest,
    ApiCreateSubmissionRequest,
)

FILE = os.sys.argv[1] if len(os.sys.argv) > 1 else "submission_lr_v2.csv"
MSG = os.sys.argv[2] if len(os.sys.argv) > 2 else "diag submit"
COMP = "nlp-getting-started"

t0 = time.time()
def ts(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

api = KaggleApi()
api.authenticate()
ts("authenticated")

with api.build_kaggle_client() as kaggle:
    request = ApiStartSubmissionUploadRequest()
    request.competition_name = COMP
    request.file_name = os.path.basename(FILE)
    request.content_length = os.path.getsize(FILE)
    request.last_modified_epoch_seconds = int(os.path.getmtime(FILE))
    ts("calling start_submission_upload ...")
    response = kaggle.competitions.competition_api_client.start_submission_upload(request)
    ts(f"got upload URL: {response.create_url[:60]}...")

    # PUT to GCS with explicit timeout; www.googleapis.com is unreachable
    # from this network, storage.googleapis.com serves the same JSON API.
    put_url = response.create_url.replace("www.googleapis.com", "storage.googleapis.com")
    ts(f"PUT file to GCS (host swapped, timeout=120s) ...")
    with open(FILE, "rb") as fp:
        r = requests.put(put_url, data=fp, timeout=(30, 120))
    ts(f"PUT status: {r.status_code}")
    if r.status_code not in (200, 201):
        print("PUT body:", r.text[:300])
        raise SystemExit(1)

    submit_request = ApiCreateSubmissionRequest()
    submit_request.competition_name = COMP
    submit_request.blob_file_tokens = response.token
    submit_request.submission_description = MSG
    ts("calling create_submission ...")
    submit_response = kaggle.competitions.competition_api_client.create_submission(submit_request)
    ts(f"submission created: ref={submit_response.ref}")
    print("message:", submit_response.message)
