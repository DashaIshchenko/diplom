import requests
from typing import Dict, Optional


class JenkinsClient:
    def __init__(self, base_url: str, username: str, api_token: str):
        self.base_url = base_url.rstrip("/")
        self.auth = (username, api_token)

    def get_build_info(self, job_name: str, build_number: int) -> Dict:
        url = f"{self.base_url}/job/{job_name}/{build_number}/api/json"
        response = requests.get(url, auth=self.auth)
        response.raise_for_status()
        return response.json()

    def get_build_log(self, job_name: str, build_number: int) -> str:
        url = f"{self.base_url}/job/{job_name}/{build_number}/consoleText"
        response = requests.get(url, auth=self.auth)
        response.raise_for_status()
        return response.text
