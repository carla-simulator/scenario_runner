class Pedestrian:
    def __init__(self, gender: str) -> None:
        self.gender = gender


class Man(Pedestrian):
    def __init__(self) -> None:
        super().__init__("male")


class Women(Pedestrian):
    def __init__(self) -> None:
        super().__init__("female")
