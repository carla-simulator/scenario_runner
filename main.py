import paho.mqtt.client as mqtt
import subprocess
import time

APP_NAME = 'avsim-carla'

broker_ip = "192.168.0.30"
class Control:
    def __init__(self):
        self.message_api = "flame/avsim/carla/mapi_set_scenario_start"
        self.mq_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1,APP_NAME, True, protocol=mqtt.MQTTv311, transport='tcp')
        self.mq_client.on_connect = self.on_mqtt_connect
        self.mq_client.on_message = self.on_mqtt_message
        self.mq_client.connect_async(broker_ip, port=1883, keepalive=60)
        self.mq_client.loop_start()
    
    def on_mqtt_connect(self, client, userData, flag, rc):
        print(f"Connected carla receiving {rc}")
        self.mq_client.subscribe(self.message_api, 0)
        
    def on_mqtt_message(self, client,userdata, msg):
        proc = subprocess.Popen(args=["python","manual_control.py", "--res","4120x1080", "--rolename=ego_vehicle"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = proc.communicate()
        print(stdout)
        if stderr:
            print("Error : ",stderr)


control = Control()
while True:
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt: # ctrl + c
        print("Exiting")
        control.mq_client.loop_stop()