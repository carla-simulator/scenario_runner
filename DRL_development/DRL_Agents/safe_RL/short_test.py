
action_dim = int(7)

if (action_dim % 2) == 1:
    action_dim -= 1

action_division = 2 / action_dim

ccc = action_dim/2

negative_action = [-1 + i * action_division for i in range(int(action_dim/2))]
positive_action = [0 + (1+i) * action_division for i in range(int(action_dim/2))]

action_space = negative_action
action_space.extend([0])
action_space.extend(positive_action)


print(minus_action)