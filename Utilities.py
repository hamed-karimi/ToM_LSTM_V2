import json
from types import SimpleNamespace

class Utilities:
    def __init__(self, ):
        self.res_folder = None
        with open('./Parameters.json', 'r') as json_file:
            self.params = json.load(json_file,
                                    object_hook=lambda d: SimpleNamespace(**d))

