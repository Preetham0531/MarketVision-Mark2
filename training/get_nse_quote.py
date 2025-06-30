from nsetools import Nse
from nsepython import nse_eq
import sys

symbol = sys.argv[1] if len(sys.argv) > 1 else 'TCS'

try:
    nse = Nse()
    print(nse.get_quote(symbol))
except Exception as e:
    print(f"nsetools failed: {e}\nTrying nsepython...")
    try:
        print(nse_eq(symbol))
    except Exception as e2:
        print(f"nsepython also failed: {e2}\nNo data could be fetched for {symbol}.") 