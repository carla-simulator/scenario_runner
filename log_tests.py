import carla
import os

client = carla.Client('127.0.0.1', 2000)

location = "{}/{}.log".format(os.getenv('SCENARIO_RUNNER_ROOT', "./"), 'RouteScenario_0')
info = client.show_recorder_file_info(location, True)


info_list = info.split("\n")
i = 0

for inform in info_list:
    # if i % 200 == 0:
    #     input()
    print(inform)
    i += 1

# info2 = client.show_recorder_file_info(location, False)
