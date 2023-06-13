import xml.etree.ElementTree as ET
import random
# 切割粒度
cut_level='Event'   # 'Act'  'Event'

cut_times = 2

addedEvent = []

class storyborad():
    def __init__(self,storyborad,eventList=None,simulation_times = 0) -> None:
        
        self.eventList = []
        if eventList:
            for event in eventList:
                self.eventList.append(event)

        self.init=init(storyborad.find('Init'))

        self.storys=[]
        for s in storyborad.findall('Story'):
            self.storys.append(story(s,self.eventList,simulation_times))



        self.story_num=len(self.storys)

        # story状态
        self.story_states=[]
        for i in range(self.story_num):
            self.story_states.append({
                    'name':self.storys[i].name,
                    'state':'run', #'run' 'end'
                })

        self.stoptrigger=storyborad.find('StopTrigger')
    
    # 本质是getRuntag
    def getSceneFile(self):


        emp_tag=ET.parse('template.xosc').getroot().find('Storyboard')

        emp_tag.append(self.init.getFullTag())

        for i in range(self.story_num):
            if self.story_states[i]['state'] == 'run':
                # if cut_level == 'Event':
                #     emp_tag.append(self.storys[i].getFullTag(eventList=event_list))
                # else:
                emp_tag.append(self.storys[i].getFullTag())

        emp_tag.append(self.stoptrigger)

        

        return emp_tag

    def update(self,resdata):
        if resdata['trigger'] =='ENDSTORYBOARD':
        # if resdata['trigger']['ENDSTORYBOARD']:
            return True
        # 更新init
        self.init.update(resdata)
        # -----------------------旧init中的action是否完成----------------------------

        # 更新story
        for i in range(self.story_num):
            key='>'+self.storys[i].name



            if resdata['blackboard'][key]:

                self.story_states[i]['state'] = 'end'

            else:
                self.storys[i].update(resdata) 

        # --------------更新stoptrigger-----------------------------

        # ---------------------根据自身状态判断是否仿真结束-------------------

        return False
    

    def flashState(self,state_dic):
        for i in range(self.story_num):
            self.story_states[i] = state_dic[str(i)]

            self.init.ifTempState = True

                      
class init():
    def __init__(self,init) -> None:
        self.meta=init
        self.ifTempState=False

        self.globalaction = init.find('Actions').find('GlobalAction')
        self.userdefinedaction = init.find('Actions').find('UserDefinedAction')
        self.controllerActions={}

        for private in init.iter('Private'):
            actor_name = private.attrib['entityRef']
            for privateaction in private.iter('PrivateAction'):
                controllerAction = privateaction.find('ControllerAction')

                if controllerAction is not None:
                    self.controllerActions[actor_name] = privateaction
                    

    def getFullTag(self):
        if not self.ifTempState:
            return self.meta
        
        else:
            emp_tag=ET.parse('template.xosc').getroot().find('Init')
            actions=emp_tag.find('Actions')
            actions.append(self.globalaction)
            
            for actor in self.init_data:
                private=ET.parse('template.xosc').getroot().find('Private')
                private.attrib['entityRef']=actor

                if actor in self.controllerActions:
                    private.append(self.controllerActions[actor])

                worldposition = private.findall('PrivateAction')[0].find('TeleportAction').find('Position').find('WorldPosition')
                worldposition.attrib['x']='{:.4f}'.format(self.init_data[actor]['world_position']['x'])
                worldposition.attrib['y']='{:.4f}'.format(self.init_data[actor]['world_position']['y'])
                worldposition.attrib['z']='{:.4f}'.format(self.init_data[actor]['world_position']['z'])
                worldposition.attrib['h']='{:.4f}'.format(-self.init_data[actor]['world_position']['yaw'])
                worldposition.attrib['p']='{:.4f}'.format(self.init_data[actor]['world_position']['pitch'])
                worldposition.attrib['r']='{:.4f}'.format(self.init_data[actor]['world_position']['roll'])

                SpeedAction = private.findall('PrivateAction')[1].find('LongitudinalAction').find('SpeedAction').find('SpeedActionTarget').find('AbsoluteTargetSpeed')
                SpeedAction.attrib['value']='{:.4f}'.format(self.init_data[actor]['speed'])

                actions.append(private)

            
            return emp_tag

            
    def update(self,resdata):
        self.init_data=resdata['actor_info']
        if 'Datetime' in resdata['blackboard']:
            self.datetime = resdata['blackboard']['Datetime']
            # print(self.datetime)

        self.ifTempState=True
        # for data in self.init_data:
        #     print(data)
        
class story():
    def __init__(self,story,eventList = None,simulation_times = 0) -> None:

        if 'name' in story.attrib:
            self.name=story.attrib['name']
        else:
            self.name=''

        self.eventList = []
        if eventList:
            for event in eventList:
                self.eventList.append(event)


        self.acts=[]
        for a in story.findall('Act'):
            self.acts.append(act(a,self.eventList,simulation_times))
        
        self.act_num=len(self.acts)

        if self.act_num == 1:
            self.cut_level='Event'
        else:
            self.cut_level='Act'

        self.simulation_times = simulation_times

        # act状态
        self.act_states=[]
        for i in range(self.act_num):
            if i==0:
                self.act_states.append({
                    'name':self.acts[i].name,
                    'state':'toadd', #'toadd' 'added' 'wait' 'run' 'end' 'trigger'
                })
            else:
                self.act_states.append({
                    'name':self.acts[i].name,
                    'state':'wait', #'wait' 'run' 'end'
                })

    def getFullTag(self):

        # 空story标签
        emp_tag=ET.parse('template.xosc').getroot().find('Story')
        emp_tag.attrib['name']=self.name

        if self.act_num == 1:
            # 唯一一个act 
            act = self.acts[0]

            emp_tag.append(act.getPartTag(self.simulation_times))

            self.act_states[0]['state'] = 'added'

        else:        
            for i in range(self.act_num):
                if self.act_states[i]['state'] == 'toadd':
                    emp_tag.append(self.acts[i].getFullTag())
                    self.act_states[i]['state'] = 'added'

                    continue

                if self.act_states[i]['state'] == 'trigger':
                    emp_tag.append(self.acts[i].getTriggerTag())
                    self.act_states[i]['state'] = 'run'

                    continue

                if self.act_states[i]['state'] == 'added':
                    emp_tag.append(self.acts[i].getFullTag())

                    continue

                if self.act_states[i]['state'] == 'wait':
                    emp_tag.append(self.acts[i].getTrigger())

                    continue

                if self.act_states[i]['state'] == 'run':
                    emp_tag.append(self.acts[i].getRunTag())   #只添加正处于run的标签
                    # emp_tag.append(self.acts[i].getFullTagWithState())  #添加所有标签,并为每个标签做状态标识

                    continue

                if self.act_states[i]['state'] == 'end':
                    continue



        return emp_tag

    def update(self,resdata):
        # 更新act的状态
        #'added' 'wait' 'run' 'end' 'trigger'

        for i in range(self.act_num):

            act = self.acts[i]
            state = self.act_states[i]['state']

            
            if state == 'wait':
                # 判断是否是被触发
                if  '(ACT)'+act.name  ==  resdata['trigger']:
                    self.act_states[i]['state'] = 'trigger'

        
            elif state == 'added':
                # 是否开始执行
                if '(ACT)'+act.name+'-START' in resdata['blackboard']:
                    # 是否结束
                    if resdata['blackboard'][act.name]:
                        # 已经结束
                        self.act_states[i]['state'] = 'end'
                    else:
                        self.act_states[i]['state'] = 'run'

                        
            elif state == 'run':
                # 判断是否结束
                if resdata['blackboard'][act.name]:          
                    self.act_states[i]['state'] = 'end' 

            act.update(resdata)


class act():
    def __init__(self,act,eventList=None,simulation_times=0) -> None:
        if 'name' in act.attrib:
            self.name=act.attrib['name']
        else:
            self.name=''

        self.eventList = []
        self.cuttedEventList=[]

        self.simulation_times = simulation_times

        unknown_events = []
        for event in act.iter('Event'):
            if event.attrib['name'] not in eventList:
                unknown_events.append(event.attrib['name'])


        if eventList:
            for event in eventList:
                for mgTag in act.iter('ManeuverGroup'):
                    if mgTag.attrib['maximumExecutionCount'] == '1':
                        for eventTag in mgTag.iter('Event'):
                            if event == eventTag.attrib['name']:
                                self.eventList.append(event)
            

            m = len(self.eventList)//cut_times
            lastOne = m + len(self.eventList)%cut_times


            for i in range(cut_times):
                if i == cut_times -1:
                    temp=[]
                    for j in range(lastOne):
                        temp.append(self.eventList[i*m+j])

                    for event in unknown_events:
                        temp.append(event)

                    self.cuttedEventList.append(temp)
                    continue
                
                temp=[]
                # if i==0:
                #     temp.append('cruiserCollisionHeroStopEvent1')
                #     temp.append('NPCChangeLane')
                for j in range(m):
                    temp.append(self.eventList[i*m+j])

                if i==0:
                    temp.remove('vehicleSpeedUp')
                
                for event in unknown_events:
                    temp.append(event)

                self.cuttedEventList.append(temp)

            self.el = []
            for i in range(self.simulation_times+1):
                for event in self.cuttedEventList[i]:
                    self.el.append(event)



        self.starttrigger=act.find('StartTrigger')
        self.stoptrigger=act.find('StopTrigger')

        # 管理mgs
        self.maneuvergroups=[]
        for mg in act.findall('ManeuverGroup'):
            self.maneuvergroups.append(maneuvergroup(mg,self.el))

        self.maneuvergroup_num=len(self.maneuvergroups)

        # maneuvergroup状态
        self.maneuvergroup_states=[]
        for i in range(self.maneuvergroup_num):
            self.maneuvergroup_states.append({
                'name':self.maneuvergroups[i].name,
                'state':'run',   # 'run' 'end'
                'maximumExecutionCount':int(self.maneuvergroups[i].maximumExecutionCount),
                'exectution_times':0
            })

    def getFullTag(self):
        # 空act标签
        emp_tag=ET.parse('template.xosc').getroot().find('Act')
        emp_tag.attrib['name']=self.name

        # 添加mg
        for i in range(self.maneuvergroup_num):
            emp_tag.append(self.maneuvergroups[i].getFullTag())
            
        emp_tag.append(self.starttrigger)
        emp_tag.append(self.stoptrigger)
        
        return emp_tag 

    def getTriggerTag(self):
        # 空act标签
        emp_tag=ET.parse('template.xosc').getroot().find('Act')
        emp_tag.attrib['name']=self.name

        # 添加mg
        for i in range(self.maneuvergroup_num):
            emp_tag.append(self.maneuvergroups[i].getFullTag())
            
        emp_tag.append(defaultstarttriiger1(self.name+'start').getTag())
        emp_tag.append(self.stoptrigger)
        
        return emp_tag


    def getRunTag(self):
        # 空act标签
        emp_tag=ET.parse('template.xosc').getroot().find('Act')
        emp_tag.attrib['name']=self.name

        # 添加mg
        for i in range(self.maneuvergroup_num):
            # 只添加正在run的mg
            if self.maneuvergroup_states[i]['state'] == 'run':
            # 根据最大执行次数选择 剩余执行次数不止一次的getFullTagWithState 只剩一次的getRunTag
                if self.maneuvergroup_states[i]['maximumExecutionCount'] > 1:
                    emp_tag.append(self.maneuvergroups[i].getFullTagWithState(self.maneuvergroup_states[i]['exectution_times']))
                else:
                    # 只剩一次只添加正在run的子标签
                    emp_tag.append(self.maneuvergroups[i].getRunTag())
            else:
                print("endtag")
            
        # starttrigger
        emp_tag.append(defaultstarttriiger1(self.name+'start').getTag())
        emp_tag.append(self.stoptrigger)
        
        return emp_tag 
    
    def getPartTag(self,simulation_times):
        # 空act标签
        emp_tag=ET.parse('template.xosc').getroot().find('Act')
        emp_tag.attrib['name']=self.name



        # 添加mg
        for i in range(self.maneuvergroup_num):
            # 只添加正在run的mg
            if simulation_times == 0:
                
                if self.maneuvergroup_states[i]['maximumExecutionCount'] > 1:
                    emp_tag.append(self.maneuvergroups[i].getFullTag())

                else:
                    # 只剩一次只添加正在run的子标签
                    emp_tag.append(self.maneuvergroups[i].getPartTag(self.el))

            else:
                if self.maneuvergroup_states[i]['state'] == 'run':
                    if self.maneuvergroup_states[i]['maximumExecutionCount'] > 1:
                        emp_tag.append(self.maneuvergroups[i].getFullTagWithState(self.maneuvergroup_states[i]['exectution_times']))
                    else:
                        # 只剩一次只添加正在run的子标签
                        emp_tag.append(self.maneuvergroups[i].getPartTag(self.el))
            
        # starttrigger
        if not simulation_times == 0:
            emp_tag.append(defaultstarttriiger1(self.name+'start').getTag())
        else:
            emp_tag.append(self.starttrigger)

        emp_tag.append(self.stoptrigger)
        
        return emp_tag 
    
    def getTrigger(self):
        # 空act标签
        emp_tag=ET.parse('template.xosc').getroot().find('Act')
        emp_tag.attrib['name']=self.name

        emp_tag.append(self.starttrigger)
        emp_tag.append(self.stoptrigger)

        return emp_tag
    
    def checkCompleteAll(self):
        for state in self.maneuvergroup_state:
            if state['exectution_times'] < state['maximumExecutionCount']:
                return False
        return True

    def update(self,resdata): 
        for i in range(self.maneuvergroup_num):
            key = self.name+'>'+self.maneuvergroups[i].name
            ifUpdateAll=False
            if self.maneuvergroup_states[i]['maximumExecutionCount'] > 1:
                if resdata['blackboard'][key+'-executions'] > 0:
                    ifUpdateAll = True

                    self.maneuvergroup_states[i]['exectution_times'] = self.maneuvergroup_states[i]['exectution_times'] + resdata['blackboard'][key+'-executions']

                    # if self.maneuvergroup_states[i]['exectution_times'] == self.maneuvergroup_states[i]['maximumExecutionCount']:
                    #     self.maneuvergroup_states[i]['state'] = 'end'
                    #     continue
            
            # if ifUpdateAll:
            #     self.maneuvergroup_states[i]['state'] = 'run'

            if self.maneuvergroup_states[i]['state'] == 'run':
                if resdata['blackboard'][key]:
                    self.maneuvergroup_states[i]['state'] = 'end'
                else:
                    self.maneuvergroups[i].update(resdata,key,ifUpdateAll)

    

class maneuvergroup():
    def __init__(self,maneuvergroup,eventlist=None) -> None:
        
        # 属性
        if 'name' in maneuvergroup.attrib:
            self.name=maneuvergroup.attrib['name']
        else:
            self.name=''

        if 'maximumExecutionCount' in maneuvergroup.attrib:
            self.maximumExecutionCount=maneuvergroup.attrib['maximumExecutionCount']
        else:
            self.maximumExecutionCount=1
        
        # 子标签
        self.actors=maneuvergroup.find('Actors')
        self.catalogreference=maneuvergroup.find('CatalogReference')
        self.maneuvers=[]
        for m in maneuvergroup.findall('Maneuver'):
            self.maneuvers.append(maneuver(m,eventlist))
        self.maneuver_num=len(self.maneuvers)

        self.maneuver_states=[]
        for i in range(self.maneuver_num):
            self.maneuver_states.append({
                'name':self.maneuvers[i].name,
                'state':'run',  #'run' 'end'
            })

    
    def getFullTag(self):
        emp_tag=ET.parse('template.xosc').getroot().find('ManeuverGroup')

        # 属性
        emp_tag.attrib['name']=self.name
        emp_tag.attrib['maximumExecutionCount']=str(self.maximumExecutionCount)

        # 子标签
        if self.actors is not None:
            emp_tag.append(self.actors)
        if self.catalogreference:
            emp_tag.append(self.catalogreference)
        for i in range(self.maneuver_num):
            emp_tag.append(self.maneuvers[i].getFullTag())
            # 如果需要对event也进行分割,则调用getPartofEvents

        return emp_tag
    

    def getFullTagWithState(self,exectution_times):
        emp_tag=ET.parse('template.xosc').getroot().find('ManeuverGroup')

        # 属性
        emp_tag.attrib['name']=self.name
        emp_tag.attrib['maximumExecutionCount']=str(self.maximumExecutionCount)

        emp_tag.attrib['exectution_times']=exectution_times

        # 子标签
        if self.actors:
            emp_tag.append(self.actors)
        if self.catalogreference:
            emp_tag.append(self.catalogreference)
        for i in range(self.maneuver_num):
            state=self.maneuver_states[i]['state']
            emp_tag.append(self.maneuvers[i].getFullTagWithState(state))

        return emp_tag        

    def getRunTag(self):
        emp_tag=ET.parse('template.xosc').getroot().find('ManeuverGroup')

        # 属性
        emp_tag.attrib['name']=self.name
        emp_tag.attrib['maximumExecutionCount']=str(self.maximumExecutionCount)

        # 子标签
        if self.actors:
            emp_tag.append(self.actors)
        if self.catalogreference:
            emp_tag.append(self.catalogreference)
        for i in range(self.maneuver_num):
            if self.maneuver_states[i]['state'] == 'run':
                emp_tag.append(self.maneuvers[i].getFullTag())

        return emp_tag
    
    def getPartTag(self,eventList):
        emp_tag=ET.parse('template.xosc').getroot().find('ManeuverGroup')

        # 属性
        emp_tag.attrib['name']=self.name
        emp_tag.attrib['maximumExecutionCount']=str(self.maximumExecutionCount)

        # 子标签
        if self.actors is not None:
            emp_tag.append(self.actors)
        if self.catalogreference:
            emp_tag.append(self.catalogreference)
        for i in range(self.maneuver_num):
            if self.maneuver_states[i]['state'] == 'run':
                emp_tag.append(self.maneuvers[i].getPartTag(eventList))

        return emp_tag

    def update(self,resdata,upperkey,ifUpdateAll):

        for i in range(self.maneuver_num):
            if ifUpdateAll:
                self.maneuver_states[i]['state'] == 'run'

            if self.maneuver_states[i]['state'] == 'run':
                key = upperkey+'>'+self.maneuvers[i].name
                if resdata['blackboard'][key]:
                    self.maneuvers[i]['state'] == 'end'
                else:
                    self.maneuvers[i].update(resdata,upperkey,ifUpdateAll)

        
class maneuver():
    def __init__(self,maneuver,eventlist=None) -> None:
        # print(eventlist)
        # 属性
        if 'name' in maneuver.attrib:
            self.name=maneuver.attrib['name']
        else:
            self.name=''    
        
        # 子标签
        self.parameterdeclaration=maneuver.find('ParameterDeclaration')
        self.events=[]
        for e in maneuver.findall('Event'):
            self.events.append(event(e))

        self.event_num=len(self.events)
        self.event_states=[]
        for i in range(self.event_num):
            state = 'toadd'
            if self.events[i].name in eventlist:
                state = 'added'
            self.event_states.append({
                'name':self.events[i].name,
                'state':state, #'toadd' 'added' 'run' 'end' 'trigger'
                'maximumExecutionCount':int(self.events[i].maximumExecutionCount),
                'exectution_times':0
            })


    def getFullTag(self):
        emp_tag=ET.parse('template.xosc').getroot().find('Maneuver')
        # 属性
        emp_tag.attrib['name']=self.name
        # 子标签
        if self.parameterdeclaration:
            emp_tag.append(self.parameterdeclaration)
        
        for i in range(self.event_num):
            emp_tag.append(self.events[i].getFullTag())

            self.event_states[i]['state'] = 'added'

        return emp_tag

    def getPartofEvents(self):
        emp_tag=ET.parse('template.xosc').getroot().find('Maneuver')

        # 属性
        emp_tag.attrib['name']=self.name
        
        # 子标签
        if self.parameterdeclaration:
            emp_tag.append(self.parameterdeclaration)

        # 决定添加哪些事件
        # wait状态添加增加已添加数量
        # added状态直接添加
        # end状态不添加
        # run状态修改事件触发条件添加

        # 已添加数量
        added_num=0

        # 添加所有剩余事件
        ifAddAll=False

        # 被触发的事件直接添加
        trigger_event=self.findTriggerEvent()
        if not trigger_event == -1 :
            emp_tag.append(self.events[trigger_event].getTriggerEvent())
            self.event_states[trigger_event]['state'] == 'added'
            added_num=added_num+1
            self.wait_event_num=self.wait_event_num-1
    
        
        for i in range(self.event_num):
            if i == trigger_event:
                continue

            if self.event_states[i]['state'] == 'end':
                continue
            
            if self.event_states[i]['state'] == 'added':
                emp_tag.append(self.events[i].getFullTag())
                continue

            if self.event_states[i]['state'] == 'run':
                emp_tag.append(self.events[i].getRunTag())
                self.event_states[i]['state'] == 'added'
                continue
        
            if self.event_states[i]['state'] == 'wait':
                # 当此事件添加达到上限
                if added_num==self.event_num_per_file:
                    if self.wait_event_num >= self.event_num_per_file:
                        emp_tag.append(self.events[i].getTrigger())
                    else:
                        ifAddAll=True
                        break
                else:
                    emp_tag.append(self.events[i].getFullTag())
                    self.wait_event_num=self.wait_event_num-1
                    added_num=added_num+1
                    self.event_states[i]['state']='added'

        if ifAddAll:
            for i in range(self.event_num):
                if self.event_states[i]['state'] == 'wait':
                    emp_tag.append(self.events[i].getFullTag())
                    self.wait_event_num=self.wait_event_num-1
                    added_num=added_num+1
                    self.event_states[i]['state']='added'

        return emp_tag

    def getFullTagWithState(self,state):
        emp_tag=ET.parse('template.xosc').getroot().find('Maneuver')

        emp_tag.attrib['state'] = state

        emp_tag.attrib['name']=self.name
        
        # 子标签
        if self.parameterdeclaration:
            emp_tag.append(self.parameterdeclaration)

        # 被触发的事件直接添加
        for i in range(self.event_num):
            state =self.event_states[i]['state']
            emp_tag.append(self.events[i].getFullTagWithState(state))

        return emp_tag        

    def getRunTag(self):
        emp_tag=ET.parse('template.xosc').getroot().find('Maneuver')

        # 属性
        emp_tag.attrib['name']=self.name

        # 子标签
        if self.parameterdeclaration:
            emp_tag.append(self.parameterdeclaration)


        for i in range(self.event_num):
            if self.event_states[i]['state'] == 'end':
                continue
            
            if self.event_states[i]['state'] == 'wait':
                emp_tag.append(self.events[i].getFullTag())

            if self.event_states[i]['state'] == 'run':
                emp_tag.append(self.events[i].getRunTag())
                continue
    
        return emp_tag
    
    def getPartTag(self,eventList):

        emp_tag=ET.parse('template.xosc').getroot().find('Maneuver')

        # 属性
        emp_tag.attrib['name']=self.name

        # 子标签
        if self.parameterdeclaration:
            emp_tag.append(self.parameterdeclaration)

        for i in range(self.event_num):
            print(self.events[i].name)
            print(self.event_states[i]['state'])

            if self.event_states[i]['state'] == 'toadd': 
                if self.events[i].name in eventList:
                    emp_tag.append(self.events[i].getFullTag())

                    self.event_states[i]['state'] = 'added'

                else:
                    emp_tag.append(self.events[i].getTrigger())

                continue
            
            if self.event_states[i]['state'] == 'added':
                emp_tag.append(self.events[i].getFullTag())

                continue

            if self.event_states[i]['state'] == 'trigger':
                emp_tag.append(self.events[i].getTriggerEvent())
                continue

            if self.event_states[i]['state'] == 'run':
                emp_tag.append(self.events[i].getRunTag())
                continue

        return emp_tag 

    def findTriggerEvent(self):
        for i in range(self.event_num):
            if self.event_states[i]['state']=='trigger':
                return i
            else:
                return -1

    def update(self,resdata,upperkey,ifUpdateAll):
        for i in range(self.event_num):
            key = upperkey+'>'+self.events[i].name
            if ifUpdateAll:
                self.event_states[i]['state'] == 'added'

            event = self.events[i]
            state = self.event_states[i]['state']


            if  '(EVENT)'+event.name  ==  resdata['trigger']:
                self.event_states[i]['state'] = 'trigger'

            elif state == 'added':
                # 是否开始执行
                
                if '(EVENT)'+self.events[i].name+'-START' in resdata['blackboard']:

                    # 是否结束
                    if resdata['blackboard'][key]:
                        # 已经结束
                        self.event_states[i]['state'] = 'end'
                    else:
                        self.event_states[i]['state'] = 'run'

            elif state == 'run':
                # 判断是否结束
                if resdata['blackboard'][key]:          
                    self.event_states[i]['state'] = 'end'

            if not self.event_states[i]['state'] == 'end':
                event.update(resdata,key,ifUpdateAll)




class event():
    def __init__(self,event) -> None:
        # 属性
        if 'name' in event.attrib:
            self.name=event.attrib['name']
        else:
            self.name=''

        if 'maximumExecutionCount' in event.attrib:
            self.maximumExecutionCount=int(event.attrib['maximumExecutionCount'])
        else:
            self.maximumExecutionCount=1

        if 'priority' in event.attrib:
            self.priority=event.attrib['priority']
        else:
            self.priority=''
        
        # 子标签
        self.starttrigger =starttrigger(event.find('StartTrigger'))
        self.actions=[]
        for ac in event.findall('Action'):
            self.actions.append(action(ac))

        self.action_num=len(self.actions)
        self.action_states=[]
        for i in range(self.action_num):
            self.action_states.append({
                'name':self.actions[i].name,
                'actionstate':'run',
                'state':{
                    
                }
            })


    def getFullTag(self):
        emp_tag=ET.parse('template.xosc').getroot().find('Event')
        # 属性
        emp_tag.attrib['name']=self.name
        emp_tag.attrib['maximumExecutionCount']=str(self.maximumExecutionCount)
        emp_tag.attrib['priority']=self.priority

        # 子标签
        emp_tag.append(self.starttrigger.getTag())
        for ac in self.actions:
            emp_tag.append(ac.getFullTag())

        return emp_tag

    def getFullTagWithState(self):
        
        emp_tag=ET.parse('template.xosc').getroot().find('Event')
        # 属性
        emp_tag.attrib['name']=self.name
        emp_tag.attrib['maximumExecutionCount']=str(self.maximumExecutionCount)
        emp_tag.attrib['priority']=self.priority

        # 子标签
        emp_tag.append(self.starttrigger.getTag())
        for i in range(self.action_num):
            state= self.action_states[i]['state']
            actionstate= self.action_states[i]['actionstate']
            emp_tag.append(self.actions[i].getFullTagWithState(actionstate,state))

        return emp_tag
    
    def getRunTag(self):
        emp_tag=ET.parse('template.xosc').getroot().find('Event')
        # 属性
        emp_tag.attrib['name']=self.name
        emp_tag.attrib['maximumExecutionCount']=str(self.maximumExecutionCount)
        emp_tag.attrib['priority']=self.priority

        # 子标签
        # emp_tag.append(self.starttrigger.getTag())
        emp_tag.append(defaultstarttriiger1('defaultstarttriiger_'+self.name).getTag())
        for i in range(self.action_num):
            if self.action_states[i]['actionstate'] == 'run':
                state= self.action_states[i]['state']
                actionstate= self.action_states[i]['actionstate']
                emp_tag.append(self.actions[i].getFullTagWithState(actionstate,state))

        return emp_tag
    
    def getTrigger(self):
        emp_tag=ET.parse('template.xosc').getroot().find('Event')
        # 属性
        emp_tag.attrib['name']=self.name
        emp_tag.attrib['maximumExecutionCount']=str(self.maximumExecutionCount)
        emp_tag.attrib['priority']=self.priority

        # 子标签
        emp_tag.append(self.starttrigger.getTag())

        return emp_tag

    def getTriggerEvent(self):
        emp_tag=ET.parse('template.xosc').getroot().find('Event')
        # 属性
        emp_tag.attrib['name']=self.name
        emp_tag.attrib['maximumExecutionCount']=str(self.maximumExecutionCount)
        emp_tag.attrib['priority']=self.priority

        # 子标签
        emp_tag.append(defaultstarttriiger1('defaultstarttriiger1').getTag())
        for ac in self.actions:
            emp_tag.append(ac.getFullTag())
        
        return emp_tag
    
    def update(self,resdata,upperkey,ifUpdateAll):
        for i in range(self.action_num):
            if ifUpdateAll:
                self.action_states[i]['actionstate'] = 'run'
                self.action_states[i]['state'] = {}


            action = self.actions[i]
            key = upperkey +'>'+action.name
            dic={}
            ifEnd = True
            if self.action_states[i]['actionstate'] == 'run':
                for index in resdata['blackboard']:

                    if index.startswith(key):
                        substr = index[len(key)+1]
                        state = resdata['blackboard'][index]
                        if state:
                            dic[substr] = 'end'
                        else:
                            ifEnd = False
                            dic[substr] = 'run'

            if ifEnd:
                self.action_states[i]['actionstate'] = 'end'

            self.action_states[i]['state'] = dic
            
class action():
    def __init__(self,action) -> None:
        if 'name' in action.attrib:
            self.name=action.attrib['name']
        else:
            self.name=''

        
        self.globalaction=action.find('GlobalAction')
        self.userdefinedaction=action.find('UserDefinedAction')

        # PrivateAction 对每个actor都要执行
        self.privateaction=action.find('PrivateAction')

        if self.privateaction is not None:
            self.states=''

            
    def getFullTag(self):
        emp_tag=ET.parse('template.xosc').getroot().find('Action')

        emp_tag.attrib['name'] = self.name

        if self.privateaction is not None:
            emp_tag.append(self.privateaction)

        
        if self.globalaction:
            emp_tag.append(self.globalaction)


        if self.userdefinedaction:
            emp_tag.append(self.userdefinedaction)

        
        return emp_tag
    
    def getFullTagWithState(self,actionstate,state):
        emp_tag=ET.parse('template.xosc').getroot().find('Action')

        emp_tag.attrib['name'] = self.name


        if self.privateaction is not None:
            emp_tag.append(self.privateaction)


        
        if self.globalaction is not None:
            emp_tag.append(self.globalaction)


        if self.userdefinedaction is not None:
            emp_tag.append(self.userdefinedaction)
 
        emp_tag.attrib['actionstate'] = actionstate
        emp_tag.attrib['state'] = str(state)

        return emp_tag
    
    # def getRunTagWithState(self,state):
    #     emp_tag=ET.parse('template.xosc').getroot().find('Action')

    #     emp_tag.attrib['name'] = self.name

    #     firstState = ''
    #     for i in state:
    #         firstState = state[i]
    #         break
            
    #     if self.privateaction:
    #         emp_tag.append(self.privateaction)
    #         if(len(state)>1):
    #             emp_tag.attrib['state'] = str(state)
    #         else:
    #             emp_tag.attrib['state'] = firstState

        
    #     if self.globalaction:
    #         emp_tag.append(self.globalaction)
    #         emp_tag.attrib['state'] = firstState


    #     if self.userdefinedaction:
    #         emp_tag.append(self.userdefinedaction)
    #         emp_tag.attrib['state'] = firstState


    #     return emp_tag

    def update(self,resdata):
        # 更新self.states
        pass

class starttrigger():
    def __init__(self,starttrigger) -> None:
        self.starttriiger=starttrigger
    
    def getTag(self):
        return self.starttriiger


    def updata():
        # 更新内部trigger
        pass
    


# 仿真时间大于0的trigger
class defaultstarttriiger1(starttrigger):
    def __init__(self,name) -> None:
        st=ET.parse('template.xosc').getroot().findall('StartTrigger')[0]
        st.find('ConditionGroup').find('Condition').attrib['name']=name

        super().__init__(st)



# 引用标签开始执行的trigger
class defaultstarttriiger2(starttrigger):
    def __init__(self,name,reftype,ref) -> None:
        st=ET.parse('template.xosc').getroot().findall('StartTrigger')[1]
        condition=st.find('ConditionGroup').find('Condition')
        condition.attrib['name']=name
        condition.find('ByValueCondition').find('StoryboardElementStateCondition').attrib['storyboardElementType']=reftype
        condition.find('ByValueCondition').find('StoryboardElementStateCondition').attrib['storyboardElementRef']=ref

        super().__init__(st)


