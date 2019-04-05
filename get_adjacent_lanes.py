

def get_adj_lanes_json():
    args = Arguments()
    client = carla.Client(args.host, int(args.port))
    client.set_timeout(25.0)
    challenge = ChallengeEvaluator(args)

    filename = os.path.join(self.root_route_file_position, 'all_towns_traffic_scenarios.json')
    world_annotations = parser.parse_annotations_file(filename)
    # retrieve routes
    # Which type of file is expected ????

    # For each of the routes to be evaluated.
    print(" all keys ", world_annotations.keys())
    list_failed = []
    for town_name in world_annotations.keys():
        if town_name != 'Town01' and  town_name != 'Town02':
            continue

        print("Town Name ", town_name)

        scenarios = world_annotations[town_name]
        for scenario in scenarios:  # For each existent scenario
            print("Scenario ", scenario['scenario_type'])
            for event in scenario["available_event_configurations"]:
                waypoint = event['transform']

                convert_waypoint_float(waypoint)
                print("Spawn ", waypoint)
                try:
                    challenge.prepare_ego_car(convert_json_to_transform(waypoint))
                except:
                    traceback.print_exc()
                    list_failed.append((waypoint, town_name))

                challenge.world.wait_for_tick()
                if 'other_actors' in event:
                    if 'left' in event['other_actors']:
                        for other_waypoint in event['other_actors']['left']:

                                list_failed.append((waypoint, town_name))
                            challenge.world.wait_for_tick()
                            print("Spawn left", other_waypoint)
                    if 'right' in event['other_actors']:
                        for other_waypoint in event['other_actors']['right']:
                            try:
                                challenge.prepare_ego_car(convert_json_to_transform(other_waypoint))
                            except:
                                traceback.print_exc()
                                list_failed.append((waypoint, town_name))
                            challenge.world.wait_for_tick()
                            print("Spawn right", other_waypoint)
                    if 'front' in event['other_actors']:
                        for other_waypoint in event['other_actors']['front']:
                            try:
                                challenge.prepare_ego_car(convert_json_to_transform(other_waypoint))
                            except:
                                traceback.print_exc()
                                list_failed.append((waypoint, town_name))
                            challenge.world.wait_for_tick()
                        print("Spawn front", other_waypoint)

