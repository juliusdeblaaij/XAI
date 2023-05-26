from EventsBroadcaster import broadcast, subscribe
from indicators.HippStringPrinter import HippStringPrinter
from indicators.IntegerCompositeIndicator import IntegerCompositeIndicator

integer_algo = IntegerCompositeIndicator()
hippe_string_printer = HippStringPrinter()
def print_addition_result(data):
    data = data.value()
    if "addition" in data:
        print(f'Addition result: {data["addition"]}')

subscribe('data_sent', print_addition_result)

print('Sending A')
broadcast('data_sent', {"A": 1})
print("Sending B")
broadcast('data_sent', {"B": 3})
broadcast('data_sent', {"hip_string": "Cool"})