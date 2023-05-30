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
                event_mes=[]
                event_mes.append(get_xml_path(act,Event))
                event_mes.append(Event)
                event_mes.append(False)
                m.append(event_mes)
            
            mg.append(m)

        act_events.append(mg)
    return act_events
    # print(act_events)



def cut_Scene(emp_Event,pop_Event):
    scene_count=2
    aver=len(emp_Event)//scene_count  
    added_event=[]
    # print(list(emp_Event[0]))
    for i in range(aver): 
        for action in pop_Event[i][1].findall('Action'):
            emp_Event[i].append(action)

        added_event.append(pop_Event[i][0])
        pop_Event[i][2]=True
    return added_event


    

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
def firstSceneFile(path,pop_Event):
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

            # 分割函数
            added_event = cut_Scene(emp_Event,pop_Event[i][j])

    print("1.生成成功")
    return scenetree,added_event

def generateInit(init,resdata):
    pass

def addEvent(events):
    added_event=[]
    for event in events:
        if not event[2]:
            added_event.append(event[1])

    return added_event


def tempSceneFile(path,added_event,resdata,events):
    template=ET.parse('template.xosc')
    temproot=template.getroot()

    print("1.生成中间场景文件")
    scenetree=ET.parse(path)
    scenefile=scenetree.getroot()

    # 生成init
    init=scenefile.find('Storyboard').find('Init')
    generateInit(init,resdata)

    # 生成act

    story=scenefile.find('Storyboard').find('Story')
    act=story.find('Act')
    # for event in act.iter('Event'):
    #     for action in event.findall('Action'):
    #         event.remove(action)
    #     for starttrigger in event.findall('StartTrigger'):
    #         event.remove(starttrigger)


    ifRunningEvent=False
    runningEventName=''
    # 
    ManeuverGroups =act.findall('ManeuverGroup')
    for i in range(len(ManeuverGroups)):
        Maneuver=ManeuverGroups[i].findall('Maneuver')
        for j in range(len(Maneuver)):
            for event in Maneuver[j].findall('Event'):
                event_name=get_xml_path(act,event)

                if event_name in added_event:
                    event_status=0      #未开始

                    event_start='(EVENT)'+event.attrib['name']+'-START'
                    if event_start in resdata['blackboard'].keys():
                        event_status=1#已开始
                    # # 事件开始
                    # # (EVENT)LeadingVehicleKeepsVelocity-START': 3.1378432661294937
                    
                    # 已结束
                    event_name=get_xml_path(story, ManeuverGroups[i])+ '>' + get_xml_path(Maneuver[j], event)
                    if resdata['blackboard'][event_name]:
                        event_status=2
                    # variable_name= get_xml_path(story, sequence) + '>' + \get_xml_path(maneuver, child) + '>' + \str(actor_id)

                    if event_status==0:
                        pass
                    elif event_status==1:
                        ifRunningEvent=True
                        runningEventName=event.attrib['name']
                        # 正在running
                        for trigger in event.findall('StartTrigger'):
                            event.remove(trigger)
                    elif event_status==2:
                        # 删除该事件
                        Maneuver[j].remove(event)
                elif event_name==resdata['triggerEvent']:
                    for e in events:
                        if e[0]==resdata['triggerEvent']:
                            e[2]=True
                            break
                    
                    triggerevent=event
                    # 对该事件添加特殊触发器
                    triggerevent.remove(triggerevent.find('StartTrigger'))
                    if ifRunningEvent:
                        starttrigger = temproot.findall('StartTrigger')[0]
                        storyboardElementStateCondition=starttrigger.find('ConditionGroup').find('Condition').find('ByValueCondition').find('StoryboardElementStateCondition')
                        storyboardElementStateCondition.attrib["storyboardElementRef"]=runningEventName
                        triggerevent.append(starttrigger)
                    else:
                        starttrigger = temproot.find('StartTrigger')[1]
                        triggerevent.append(starttrigger)
                else:
                    added_event_new=addEvent(events)
                    if event_name not in added_event_new:
                        for action in event.findall('Action'):
                            event.remove(action)
    
    scenetree.write('temp2.xosc')
    



def sendfile(scenefile,ip,port):
    print("-----------分发场景文件---------")
    data = pickle.dumps(scenefile)
    data_size=len(data)

    client=socket.socket()
    client.connect((ip, port))

    f= struct.pack("l",data_size)
    
    client.send(f)
    
    client.sendall(data)
    print("-----------分发成功---------")

    client.close()

def listen():
    print("2.等待返回仿真结果")
    server = socket.socket()         
    server.bind(('127.0.0.1', 9002)) 
    server.listen() 
    # while True:
    print("start.......")
    sock,adddr = server.accept()
    d = sock.recv(struct.calcsize("l"))
    total_size = struct.unpack("l",d)
    num  = total_size[0]//1024
    data = b''
    for i in range(num):
        data += sock.recv(1024)
    data += sock.recv(total_size[0]%1024)

    resdata=pickle.loads(data)

    print("2.得到返回结果")
    sock.close()
    return resdata
    


if __name__=="__main__":
    path=r'F:\C\V6\scenario_runner\srunner\examples\FollowLeadingVehicle.xosc'
    # filetemppath=r'D:\work\ADS\carla\runner\scenario_runner\srunner\examples\temp.xosc'


    

    ifEnd=False

    tree = ET.parse(path)
    root = tree.getroot()
    # 弹出所有事件
    events=popEvents(root)
    # 组装第一个场景文件
    scenefile,added_event=firstSceneFile(path,events)
    # 发送
    sendfile(scenefile=scenefile,ip='127.0.0.1',port=9001)
    # 接受返回状态
    resdata=listen()
    
    # print(resdata)
    # while not ifEnd:
    #     scenefile=tempSceneFile(path,added_event,resdata,events)
    #     sendfile(scenefile=scenefile,ip='127.0.0.1',port=9001)
    #     resdata=listen()
    
    scenefile=tempSceneFile(path,added_event,resdata,events)




    