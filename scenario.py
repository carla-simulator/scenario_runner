import xml.etree.ElementTree as ET
from srunner.scenarios.open_scenario import get_xml_path
import socket
import pickle
import json
import struct

# 场景分割
# 考虑将一个场景文件切割成两个子文件，每个子文件携带部分事件，并携带所有触发器，切割成的子文件并不是实际给仿真器的文件
# 具体流程为：
# 1.根据某种原则（简单的如对半分）将场景文件分成多个子文件，每个文件携带所有触发器以及部分事件
# 2.先将第一个场景文件发送给runner
# 3.监听runner的返回信息
# 4.解析返回信息，依据解析结果进行
#   1）确定下一个合并的场景文件（依据事件触发）
#   2）对于要保留未执行的事件直接移植
#   3）对于要保留且已经执行的事件做转化
#   4）根据返回帧的结果设置物体的初始状态。
# 由此生成下一个场景文件

# 对于每个场景文件，定义一个对象




# 假设只有一个act
def popEvents(scene_xml):
    act=scene_xml.find('Storyboard').find('Story').find('Act')
    act_events=[]
    for ManeuverGroup in act.findall('ManeuverGroup'):
        mg=[]
        for Maneuver in ManeuverGroup.findall('Maneuver'):
            m=[]
            for Event in Maneuver.findall('Event'):
                e=Event
                m.append(e)
            
            mg.append(m)

        act_events.append(mg)
    return act_events
    # print(act_events)



def cut_Scene(emp_Event,pop_Event):
    scene_count=2
    aver=len(emp_Event)//scene_count  
    # print(list(emp_Event[0]))
    for i in range(aver):        
        emp_Event[i].append(pop_Event[i].find('Action'))


    

# def emptySceneList():
#     scenes=[]
#     for i in range(scene_count):
#         scenes.append([])

#     return scenes
    
# def cuttedSceneList(Scene_xml):
#     Scenelist=emptySceneList()
#     for maneuverGroup in Scene_xml.iter('Act'):
#         mg=[]
#         for 
#         maneuver=maneuverGroup.find('Maneuver')
#         event_list=maneuver.findall('Event')

#         aver=len(event_list)//scene_count  
#         tail_event=len(event_list)%scene_count

#         for i in range(scene_count):
#             for j in range(aver):
#                 Scenelist[i].append(event_list[i*aver+j])

#         for i in range(tail_event):
#             Scenelist[scene_count-1].append(event_list[len(event_list)-i-1])
#     return Scenelist

def Actors(Scene_xml):
    return Scene_xml.find('Entities').findall('ScenarioObject')
        
# 生成第一个场景文件
def firstSceneFile(pop_Event):
    print("1.生成第一个场景文件")
    scenetree=ET.parse(path)
    scenefile=scenetree.getroot()
    act=scenefile.find('Storyboard').find('Story').find('Act')

    for event in act.iter('Event'):
        for action in event.findall('Action'):
            event.remove(action)

    ManeuverGroups =act.findall('ManeuverGroup')
    for i in range(len(ManeuverGroups)):
        Maneuver=ManeuverGroups[i].findall('Maneuver')

        for j in range(len(Maneuver)):
            emp_Event = Maneuver[j].findall('Event')
            pop_Event[i][j]

            # 分割函数
            cut_Scene(emp_Event,pop_Event[i][j])

    print("1.生成成功")
    return scenetree

# tree.write(r"D:\work\ADS\esmini\esmini-demo_win\esmini-demo\resources\xosc\temp.xosc")



def sendfile(scenefile):
    print("-----------分发场景文件---------")
    data = pickle.dumps(scenefile)
    data_size=len(data)

    client=socket.socket()
    client.connect(('127.0.0.1', 9001))

    f= struct.pack("l",data_size)
    
    client.send(f)
    
    client.sendall(data)
    print("-----------分发成功---------")

    client.close()

if __name__=="__main__":
    path=r'D:\work\ADS\carla\runner\scenario_runner\srunner\examples\FollowLeadingVehicle.xosc'
    filetemppath=r'D:\work\ADS\carla\runner\scenario_runner\srunner\examples\temp.xosc'

    tree = ET.parse(path)
    root = tree.getroot()
    # 弹出所有事件
    events=popEvents(root)
    # 组装第一个场景文件
    scenefile=firstSceneFile(events)

    sendfile(scenefile=scenefile)



    