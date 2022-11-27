from typing import Union
import live
from fastapi import FastAPI
from simplecoremidi import send_midi
import time
app = FastAPI()

set = live.Set()
set.scan(scan_clip_names=True, scan_devices=True)

limit_left = False
limit_right = False
limit_front = False

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/health")
def health_check():
    return {"msg": "good"}

@app.post("/set_pan")
def set_pan(pan: float):
    send_midi((0xb0, 80, (pan + 1)/2 * 128))
    print(pan)
    center_left = (-min(0, pan)*1.2) * 128
    send_midi((0xb0, 72, center_left))
    center_right = (max(0,pan)*1.2) * 128
    send_midi((0xb0, 73, center_right))
    
    return {"msg": "good"}

@app.post("/set_volume")
def set_volume(volume:float):
    global limit_front

    volume = max(min(volume,1),0) * 128

    msg = (0xb0, 79, volume)
    send_midi(msg)    

    if volume > 128 * 0.81 and not limit_front:
        send_midi((0x90,49, 98))
        limit_front = True
    if volume < 128 * 0.81 and limit_front:
        send_midi((0x80,49,0))
        limit_front = False

    return {"msg": "good"}

@app.post('/left_arm')
def left_arm(distance: float, angle: float):
    print(distance, angle)
    #msg = (0xb0, 75, min(distance * 128, 128))
    #send_midi(msg)

    #msg = (0xb0, 74, (angle + 1) * 64)
    #send_midi(msg)

    msg = (0xb0, 82, distance * (angle + 1) * 64)
    send_midi(msg)

@app.post('/right_arm')
def right_arm(distance: float, angle: float):
    msg = (0xb0, 81, distance * (angle + 1) * 64)
    send_midi(msg)

@app.post('/activity_level')
def activity_level(first_order: float, second_order: float):
    send_midi((0xb0, 74, first_order))
    send_midi((0xb0, 75, second_order))

@app.post('/look_front' )
def isfacelookingfront(look_front: bool):
    if look_front:
        send_midi((0xb0, 76, 0))
    else:
        send_midi((0xb0, 76, 127))

@app.post('/right_shoulder')
def max_right(right_shoulder: float):
    global limit_right
    if right_shoulder < 0.1 and not limit_right:
        send_midi((0x90, 47, 98))
        limit_right = True
    if right_shoulder > 0.1 and limit_right:
        send_midi((0x80, 47, 0))
        limit_right = False

@app.post('/left_shoulder')
def max_left(left_shoulder: float):
    global limit_left
    if left_shoulder > 0.9 and not limit_left:
        send_midi((0x90, 48, 98))
        limit_left = True
    if left_shoulder < 0.9 and limit_left:
        send_midi((0x80, 48, 0))
        limit_left = False  

@app.post("/trigger_left_arm_acc")
def trigger_left_arm_acc(speed:float, acc: float):
    if abs(speed) > 0.5:
        send_midi((0x90, 53, 98))
        send_midi((0x80, 53, 0))

@app.post("/trigger_right_arm_acc")
def trigger_right_arm_acc(speed:float, acc: float):
    if abs(speed) > 0.5:
        send_midi((0x90, 54, 98))
        send_midi((0x80, 54, 0))

@app.post("/start_play")
def start_play():
    set.play(reset=True)

@app.post("/end_play")
def end_play():
    set.stop()

@app.post("/unpause_play")
def pause_play():
    set.play()

@app.post("/test_midi")
def test_midi():        
    send_midi((0x90, 0x3c, 0x40))
