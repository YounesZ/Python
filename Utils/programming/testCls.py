class test:

    def __init__(self):
        self.player = {'hand': ['1_spade', '4_diamond'], 'value': 5}
        self.dealer = {'hand': ['10_spade', '4_diamond'], 'value': 14}

    def update_val(self, person, newValue):
        vars()["self."+person+"['value']"]    =   newValue
        print(vars()["self."+person+"[value]"])

