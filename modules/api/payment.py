from abc import abstractmethod
from fastapi import HTTPException, Request
from fastapi.security.api_key import APIKeyHeader
import requests
import psutil

HEADER_NAME = "Authorization"
API_TOPUP_URL="https://tecky.io/subscription/topup"
SERVICE_API_KEY="HtRjFlAInENAA2atZjEGLHUer5k4D1GeCROwZ5VI/l4="
SERVER_BUSY_ERROR_CODE=503

class Payment():
    def __init__(self,api_key: str):
        self.api_key = api_key
        self.route_cost={
            "/sdapi/v1/txt2img":10,
            "/sdapi/v1/img2img":10
        }
        
    def demo_available(self):
        from modules.shared import state
        if state.job_count>0:
            return False
        if psutil.cpu_percent()>80:
            return False
        return True
    
    @abstractmethod
    def pre_payment_handling(self,endpoint):
        pass
    @abstractmethod
    def post_payment_handling(self,endpoint,method):
        pass
    
class TeckyPayment(Payment):
    def __init__(self,api_key: str):
        super().__init__(api_key)
    def check_balance(self):
        url = f'https://api.tecky.io/api/v1/subscription/{self.api_key}'
        headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {SERVICE_API_KEY}'
        }
        response = requests.get(url, headers=headers)
        balance_info = response.json()
        balance=balance_info["subscription"]["credits"]
        return balance
    def pay(self,cost,remark):
        url = f"https://api.tecky.io/api/v1/service-usage/{self.api_key}"
        headers = {
            "Accept": "application/json",
            'Authorization': f'Bearer {SERVICE_API_KEY}',
            "Content-Type": "application/json",
        }
        data = {
            "service": "/api/sd",
            "cost": cost,
            "remark": remark,
        }
        res = requests.post(url, headers=headers, json=data)
        return res.json()
    def pre_payment_handling(self, endpoint):
        if endpoint=="/sdapi/v1/progress":
            return True
        if self.demo_available():
            return True
        if self.api_key is None:
            raise HTTPException(status_code=SERVER_BUSY_ERROR_CODE, \
                detail=f"Server is busy. Please try again later. Or you can get an API key from {API_TOPUP_URL}")
        balance=self.check_balance()
        if endpoint in self.route_cost:
            if balance<self.route_cost[endpoint]:
                raise HTTPException(status_code=SERVER_BUSY_ERROR_CODE, \
                    detail=f"Insufficient balance. Please top up your account. Or you can get an API key from {API_TOPUP_URL}")
        return True
    def post_payment_handling(self,endpoint,method):
        if self.api_key is None:
            return
        try:
            if method=="POST":
                if endpoint in self.route_cost:
                    self.pay(self.route_cost[endpoint],f"{endpoint} [{method}] Called")
        except Exception as e:
            print(e)
            pass
