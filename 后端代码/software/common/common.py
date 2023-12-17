from django.http import JsonResponse
import json
from datetime import datetime
from django.http import HttpRequest
from rest_framework.views import APIView
from rest_framework.response import Response
from utils.tool import base64TwoPIL, PILTwobase64, tool
from utils.detection import Net, addZeroPadding, postprocess

# 处理前端传回的图片
class SARView(APIView):
    def get(self, request):
        data = {"test": "shineahead", "status": 20000}
        # 允许前端请求跨域
        headers = {
            "Access-Control-Allow-Origin": "*",
        }
        return Response(data=data, headers=headers, status=201)
    def post(self, request):
        print("POST请求")
        # 允许前端请求跨域
        headers = {
            "Access-Control-Allow-Origin": "*",
        }
        data = {}
        # 使用json来解析字符串列表
        arr = json.loads(request.data["data"])
        img1, img2 = base64TwoPIL(arr[0]), base64TwoPIL(arr[1])
        res = tool(img1, img2, "res.bmp")
        res = PILTwobase64(res)
        data['base'] = res
        print("发送成功----------------")
        return Response(data=data, headers=headers, status=201)

