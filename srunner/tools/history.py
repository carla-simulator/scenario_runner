import py_trees

class history():
    def __init__(self,name,item_name) -> None:

        self.name = name
        self.item_name = item_name
    
    def createHistory(self):
        if not self.name in py_trees.blackboard.Blackboard().dict():
            py_trees.blackboard.Blackboard().set(self.name,{})

        if not self.item_name in py_trees.blackboard.Blackboard().get(self.name):
            py_trees.blackboard.Blackboard().get(self.name)[self.item_name] = [0,0]

        py_trees.blackboard.Blackboard().get(self.name)[self.item_name][0] = py_trees.blackboard.Blackboard().get(self.name)[self.item_name][0]+py_trees.blackboard.Blackboard().get(self.name)[self.item_name][1]
        py_trees.blackboard.Blackboard().get(self.name)[self.item_name][1] = 0 


        

    def getHistory(self):    
        return py_trees.blackboard.Blackboard().get(self.name)[self.item_name][0] +py_trees.blackboard.Blackboard().get(self.name)[self.item_name][1]
    
    def getHistorydata(self,index):
        return py_trees.blackboard.Blackboard().get(self.name)[self.item_name][index]
    
    def setHistory(self,data):


        py_trees.blackboard.Blackboard().get(self.name)[self.item_name][1] = data

    def setHistory2(self,data1,data2):
        py_trees.blackboard.Blackboard().get(self.name)[self.item_name] = [data1,data2]

    def deleteHistory(self):
        py_trees.blackboard.Blackboard().dict().get(self.name).pop(self.item_name)


class startpointhistory():
    def __init__(self,name,item_name) -> None:

        self.name = name
        self.item_name = item_name


    def initialize(self):
        if not self.name in py_trees.blackboard.Blackboard().dict():
            py_trees.blackboard.Blackboard().set(self.name,{})

    def checkStartpoint(self):
        

        return self.item_name in py_trees.blackboard.Blackboard().get(self.name)
    
    def createstartpoint(self,startpoint):


        py_trees.blackboard.Blackboard().get(self.name)[self.item_name] = startpoint



    def getstartpoint(self):
        return py_trees.blackboard.Blackboard().get(self.name)[self.item_name]




    



class waypointhistory():
    def __init__(self,name,item_name) -> None:

        self.name = name
        self.item_name = item_name

    def createHistory(self):
        if not self.name in py_trees.blackboard.Blackboard().dict():
            py_trees.blackboard.Blackboard().set(self.name,{})

    def checkwaypoint(self):
        return self.item_name in py_trees.blackboard.Blackboard().get(self.name)

    def setwaypoint(self,waypoint):
        py_trees.blackboard.Blackboard().get(self.name)[self.item_name] = waypoint

    def getwaypoint(self):
        return py_trees.blackboard.Blackboard().get(self.name)[self.item_name]

    def deletewaypoint(self):
        py_trees.blackboard.Blackboard().dict().get(self.name).pop(self.item_name)

