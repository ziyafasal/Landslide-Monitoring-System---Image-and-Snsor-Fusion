# This is to check if its predicting correctly without the input from the sensor or camera. 
# Value can be changed to check each condition.


import requests
import time
import random

URL = "http://ipaddress:8000/sensors"
PREDICT_URL = "http://ipaddress:8000/predict"

while True:
    data = {
        "rain": random.uniform(9, 90),
        "moisture": random.uniform(10, 90),
        "vibration": random.uniform(1,3),
        "lat":11.987690,
        "lon" :75.381520,
        "altitude" : 31.4,
        "satellites":8
    }

    try:
        res = requests.post(URL, json=data)
        print("Sent:", data, "| Response:", res.json())
    except Exception as e:
        print("Error:", e)

    time.sleep(2)
    
    
    with open("crack.jpeg", "rb") as f:
        res = requests.post(PREDICT_URL, data=f.read())

    print(res.json())
    time.sleep(1)

       