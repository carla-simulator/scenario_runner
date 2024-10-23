"""CARLA blueprints for cars, pedestrians, etc."""

#: Mapping from current names of blueprints to ones in old CARLA versions.
#:
#: We provide a tuple of old names in case they change more than once.
oldBlueprintNames = {
    "vehicle.dodge.charger_police": ("vehicle.dodge_charger.police",),
    "vehicle.lincoln.mkz_2017": ("vehicle.lincoln.mkz2017",),
    "vehicle.mercedes.coupe": ("vehicle.mercedes-benz.coupe",),
    "vehicle.mini.cooper_s": ("vehicle.mini.cooperst",),
    "vehicle.ford.mustang": ("vehicle.mustang.mustang",),
}

## Vehicle blueprints

#: blueprints for cars
carModels = [
    'vehicle.taxi.ford',
    'vehicle.dodgecop.charger',
    'vehicle.fuso.mitsubishi',
    'vehicle.ue4.chevrolet.impala',
    'vehicle.ue4.ford.mustang',
    'vehicle.ue4.mercedes.ccc',
    'vehicle.ue4.bmw.grantourer',
    'vehicle.ue4.ford.crown',
    'vehicle.ue4.audi.tt',
    'vehicle.dodge.charger',
    'vehicle.nissan.patrol',
    'vehicle.mini.cooper',
    'vehicle.lincoln.mkz'
]

#: blueprints for trucks
truckModels = [
    'vehicle.carlacola.actors',
    'vehicle.firetruck.actors',
    'vehicle.sprinter.mercedes',
    'vehicle.ambulance.ford',
]

## Prop blueprints

#: blueprints for trash cans
trashModels = [
    "static.prop.dumpster",
]

#: blueprints for traffic cones
coneModels = [
    "static.prop.constructioncone",
    "static.prop.trafficcone01",
    "static.prop.trafficcone02",
]

#: blueprints for road debris
debrisModels = [
    "static.prop.dirtdebris01",
    "static.prop.dirtdebris02",
    "static.prop.dirtdebris03",
]

#: blueprints for vending machines
vendingMachineModels = [
    "static.prop.vendingmachine",
]

#: blueprints for chairs
chairModels = [
    "static.prop.plasticchair",
]

#: blueprints for bus stops
busStopModels = [
    "static.prop.busstop",
]

#: blueprints for roadside billboards
advertisementModels = [
    "static.prop.advertisement",
    "static.prop.streetsign",
    "static.prop.streetsign01",
    "static.prop.streetsign04",
]

#: blueprints for pieces of trash
garbageModels = [
    "static.prop.colacan",
    "static.prop.garbage01",
    "static.prop.garbage02",
    "static.prop.garbage03",
    "static.prop.garbage04",
    "static.prop.garbage05",
    "static.prop.garbage06",
    "static.prop.plasticbag",
    "static.prop.trashbag",
]

#: blueprints for containers
containerModels = [
    "static.prop.container",
    "static.prop.clothcontainer",
    "static.prop.glasscontainer",
]

#: blueprints for tables
tableModels = [
    "static.prop.table",
    "static.prop.plastictable",
]

#: blueprints for traffic barriers
barrierModels = [
    "static.prop.streetbarrier",
    "static.prop.chainbarrier",
    "static.prop.chainbarrierend",
]

#: blueprints for flowerpots
plantpotModels = [
    "static.prop.plantpot01",
    "static.prop.plantpot02",
    "static.prop.plantpot03",
    "static.prop.plantpot04",
    "static.prop.plantpot05",
    "static.prop.plantpot06",
    "static.prop.plantpot07",
    "static.prop.plantpot08",
]

#: blueprints for mailboxes
mailboxModels = [
    "static.prop.mailbox",
]

#: blueprints for garden gnomes
gnomeModels = [
    "static.prop.gnome",
]

#: blueprints for creased boxes
creasedboxModels = [
    "static.prop.creasedbox01",
    "static.prop.creasedbox02",
    "static.prop.creasedbox03",
]

#: blueprints for briefcases, suitcases, etc.
caseModels = [
    "static.prop.travelcase",
    "static.prop.briefcase",
    "static.prop.guitarcase",
]

#: blueprints for boxes
boxModels = [
    "static.prop.box01",
    "static.prop.box02",
    "static.prop.box03",
]

#: blueprints for benches
benchModels = [
    "static.prop.bench01",
    "static.prop.bench02",
    "static.prop.bench03",
]

#: blueprints for barrels
barrelModels = [
    "static.prop.barrel",
]

#: blueprints for ATMs
atmModels = [
    "static.prop.atm",
]

#: blueprints for kiosks
kioskModels = [
    "static.prop.kiosk_01",
]

#: blueprints for iron plates
ironplateModels = [
    "static.prop.ironplank",
]

#: blueprints for traffic warning signs
trafficwarningModels = [
    "static.prop.trafficwarning",
]

## Walker blueprints

#: blueprints for pedestrians
walkerModels = [
    "walker.pedestrian.0015",
    "walker.pedestrian.0016",
    "walker.pedestrian.0017",
    "walker.pedestrian.0018",
    "walker.pedestrian.0019",
    "walker.pedestrian.0020",
    "walker.pedestrian.0021",
    "walker.pedestrian.0022",
    "walker.pedestrian.0023",
    "walker.pedestrian.0024",
    "walker.pedestrian.0025",
    "walker.pedestrian.0026",
    "walker.pedestrian.0027",
    "walker.pedestrian.0028",
    "walker.pedestrian.0029",
    "walker.pedestrian.0030",
    "walker.pedestrian.0031",
    "walker.pedestrian.0032",
    "walker.pedestrian.0033",
    "walker.pedestrian.0034",
    "walker.pedestrian.0035",
    "walker.pedestrian.0036",
    "walker.pedestrian.0037",
    "walker.pedestrian.0038",
    "walker.pedestrian.0039",
    "walker.pedestrian.0040",
    "walker.pedestrian.0041",
    "walker.pedestrian.0042",
    "walker.pedestrian.0043",
    "walker.pedestrian.0044",
    "walker.pedestrian.0045",
    "walker.pedestrian.0046",
    "walker.pedestrian.0047",
    "walker.pedestrian.0048",
    "walker.pedestrian.0049",
    "walker.pedestrian.0050",
    "walker.pedestrian.0051",
]
