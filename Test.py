from EventsBroadcaster import broadcast
from indicators.HippStringPrinter import HippStringPrinter
from indicators.IntegerCompositeIndicator import IntegerCompositeIndicator

integer_algo = IntegerCompositeIndicator()
hippe_string_printer = HippStringPrinter()

print('Sending A')
broadcast('data_sent', {"A": 1})
print("Sending B")
broadcast('data_sent', {"B": 3})
