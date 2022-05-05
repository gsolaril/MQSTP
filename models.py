import numpy, pandas
from datetime import datetime
from ProgBar import ProgBar
from time import time
from enum import Enum

#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#
 
class Symbol:
    
    def __init__(self, quote: str, point: float, value: float, minDist: float,
        digits: int = 5, minSize: float = 0.01, maxSize = 100, stepSize = 0.01,
        commission: int = 7, leverage: int = 100):

        self.minDist, self.minSize, self.maxSize = point * minDist, minSize, maxSize
        self.quote, self.digits, self.point, self.value = quote, digits, point, value
        self.stepSize, self.commission, self.leverage = stepSize, commission, leverage
        self.__ = None

    def round(self, value, size: bool = False) -> float:

        if not size:
            value = numpy.round(value / self.point) * self.point
            return numpy.around(value, decimals = self.digits)
        value = numpy.round(value / self.stepSize) * self.stepSize
        return min(max(self.minSize, numpy.around(value, 2)), self.maxSize)

    def json(self, outer = False) -> dict:

        if outer: return {self.quote: dict(
            point = self.point, value = self.value, minDist = self.minDist,
            digits = self.digits, stepSize = self.stepSize, minSize = self.minSize,
            maxSize = self.maxSize, commission = self.commission, leverage = self.leverage)}
        else: return dict(quote = self.quote,
            point = self.point, value = self.value, minDist = self.minDist,
            digits = self.digits, stepSize = self.stepSize, minSize = self.minSize,
            maxSize = self.maxSize, commission = self.commission, leverage = self.leverage)

    def __setattr__(self, name, value):
        
        if (dir(self)[0] == "__"): return
        else: object.__setattr__(self, name, value)

#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

class Candle:

    labels = {label[0].upper(): label for label in \
        ["time", "previous close", "open", "high", "low", "close", "spread", "volume"] }
    errorLabels = "\"row\" must at least keep the following (correctly) labelled values: "
    errorDFStruct = "Invalid DataFrame. Must be tick (TAB) or candle (TOHLCVS)."
    errorDFSize = "\"rows\" must be a DataFrame with at least 2 rows."

    def __init__(self, row: pandas.Series):

        ref = [f"{char} ({name})" for char, name in self.labels.items()]
        try: 
            if not ("P" in row): row["P"] = row["O"]
            if not ("T" in row): row["T"] = row.name
            assert set("TPOHLCSV").issubset(row.index)
        except: raise AssertionError(self.errorLabels + ", ".join(ref))
        
        T, P, O, H, L, C, S, V = [row[label] for label in list("TPOHLCSV")]
        assert isinstance(T, datetime), "Time values must be of \"datetime\" format."
        correctPrices = [isinstance(x, float) and (x > 0) for x in [P, O, H, L, C, S]]
        assert all(correctPrices), "All price and spread values must be positive floats."
        assert isinstance(V, (int, float)) and (V > 0), "Volumes must be of int/float format."
        
        self.open, self.high, self.low, self.close = O, H, L, C
        self.time, self.spread, self.volume, self.prev = T, S, V, P
        self.bid, self.ask = self.close, self.close + self.spread
        self.__ = None
    
    def crosses(self, value: float) -> bool:
        
        if (value == None): return False
        assert isinstance(value, (int, float)) and (value > 0), \
          "Invalid price value being compared to candle."
        crossed = (self.low <= value <= self.high)
        gapped_below = (self.prev <= value <= self.low)
        gapped_above = (self.high <= value <= self.prev)
        return crossed or gapped_below or gapped_above

    @classmethod
    def From(cls, data: pandas.DataFrame):

        try: assert (len(data.index) >= 2)
        except: raise AssertionError(cls.errorDFSize)
        prevRow, lastRow = data.iloc[-2], data.iloc[-1]
        if set(data.columns).issuperset(set("AB")):
            a, b, P = *lastRow, prevRow["B"]
            O, H, L, C, V, S = b, b, b, b, 1, a - b
        if set(data.columns).issuperset(set("OHLCVS")):
            O, H, L, C, V, S = lastRow[list("OHLCVS")]
            P = prevRow["C"]
        try: return Candle(pandas.Series(index = list("TPOHLCVS"),
                     data = [lastRow.name, P, O, H, L, C, V, S]))
        except: raise AssertionError(cls.errorDFStruct)

    def __setattr__(self, name, value):

        if (dir(self)[0] == "__"): return
        else: object.__setattr__(self, name, value)

#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

class Trading:

    @staticmethod
    def isPrice(value, z: bool = False):
        if not isinstance(value, (list, tuple)): value = [value]
        if z: cond = lambda x: isinstance(x, (int, float)) and (x >= 0)
        else: cond = lambda x: isinstance(x, (int, float)) and (x > 0)
        assert all(map(cond, value)), "Prices must be positive floats."

    class Side(Enum):
        BUY, SELL = +1, -1
        @classmethod
        def check(cls, value): assert (value in [cls.BUY, cls.SELL]), \
            "\"side\" must be one of the following: %s" % [f"Trading.Side.{x.name}" for x in list(cls)]

    class Profit(Enum):
        POINTS, CURRENCY = 1, 2
        @classmethod
        def check(cls, value): assert (value in [cls.POINTS, cls.CURRENCY]), \
             "\"unit\" must be one of the following: %s" % [f"Trading.Profit.{x.name}" for x in list(cls)]

    class nullModifier:
        
        def __init__(self): pass
        def analyze(self, data: pandas.DataFrame, type: object, json: dict): return None

    class nullCloser:
        
        def __init__(self): pass
        def analyze(self, data: pandas.DataFrame, type: object, json: dict): return None, None
             
    def __init__(self, symbol, side, ST, OP, ticket, size, mods, cause, strict):

        self.Side.check(side)
        assert isinstance(ST, datetime), "\"ST\" must be an instance of \"datetime\"."
        assert isinstance(symbol, Symbol), "\"symbol\" must be an instance of \"Symbol\" module."
        assert isinstance(cause, str), "\"cause\" must be a string not longer than 64 characters."
        assert isinstance(ticket, int), f"\"ticket\" must be a positive integer."
        assert isinstance(mods, int), f"\"mods\" must be a positive integer."
        self.symbol, self.ticket, self.cause = symbol, ticket, cause
        self.size = self._checkSize(size = size, strict = False)
        self.isPrice(OP, z = True)
        
        self.side, self.mods, self.strict = side, mods, strict 
        self.ST, self.OP = ST, self.symbol.round(OP)
        self.BLOCKED = None

    def _checkSize(self, size: float, strict: bool) -> float:

        minSize, maxSize = self.symbol.minSize, self.symbol.maxSize
        error = f"\"size\" must be between {minSize} and {maxSize}."
        assert isinstance(size, (int, float)), error
        invalid = not (minSize <= size <= maxSize)
        if invalid and strict: raise AssertionError(error)
        else: return self.symbol.round(size, size = True)

    def __setattr__(self, name, value):

        if (dir(self)[0] == "BLOCKED"): return
        else: object.__setattr__(self, name, value)

#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

class Result(Trading):

    def __init__(self, ticket: int, symbol: Symbol, side: Trading.Side, size: float,
                 ST: datetime, OT: datetime, CT: datetime, cause: str, result: str,
                 OP: float, CP: float, WP: float, mods: int):

        self.isPrice([OP, CP, WP], z = False)
        assert isinstance(result, str), "\"result\" must be a string."
        assert isinstance(OT, datetime), "\"OT\" must be a \"datetime\" instance."
        assert isinstance(CT, datetime), "\"CT\" must be a \"datetime\" instance."
        super().__init__(symbol, side, ST, OP, ticket, size, mods, cause, True)
        del self.strict, self.BLOCKED
        self.OT, self.CT, self.result = OT, CT, result
        self.CP, self.WP = map(self.symbol.round, [CP, WP])
        self.BLOCKED = None

    def gain(self, unit: Trading.Profit) -> float:

        self.Profit.check(unit)
        delta = (self.CP - self.OP) * self.side.value
        points = numpy.round(delta / self.symbol.point)
        if (unit == self.Profit.POINTS): return int(points)
        return numpy.round(self.size * self.symbol.value * points * 100) / 100

    def sink(self, unit: Trading.Profit) -> float:

        self.Profit.check(unit)
        delta = (self.WP - self.OP) * self.side.value
        points = numpy.round(delta / self.symbol.point)
        if (unit == self.Profit.POINTS): return int(points)
        return numpy.round(self.size * self.symbol.value * points * 100) / 100

    def json(self) -> dict: return dict(
        ticket = self.ticket, symbol = self.symbol.quote, side = self.side.name,
        ST = self.ST, OT = self.OT, CT = self.CT, OP = self.OP, CP = self.CP, WP = self.WP,
        size = self.size, mods = self.mods, cause = self.cause, result = self.result,
        gainPTS = self.gain(self.Profit.POINTS), gainUSD = self.gain(self.Profit.CURRENCY),
        sinkPTS = self.sink(self.Profit.POINTS), sinkUSD = self.sink(self.Profit.CURRENCY))

#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

class Trade(Trading):

    def __init__(self, ticket: int, symbol: Symbol, side: Trading.Side, size: float, mods: int,
        OT: datetime, ST: datetime, OP: float, SL: float, TP: float, cause: str, strict: bool,
        closer: object, modifier: object):

        self.isPrice(OP)
        assert isinstance(OT, datetime), "\"OT\" must be a \"datetime\" instance."
        assert hasattr(closer, "analyze"), "Trade \"closer\" must be a TradeCloser object."
        assert hasattr(modifier, "analyze"), "Trade \"modifier\" must be a TradeModifier object."
        super().__init__(symbol, side, ST, OP, ticket, size, mods, cause, strict)
        del self.BLOCKED
        self.SL = None    ;  self._modifySL(SL)
        self.TP = None    ;  self._modifyTP(TP)

        self.OT, self.closer, self.modifier = OT, closer, modifier
        self.compare = min if (side == Trading.Side.BUY) else max
        self.WP, self.actual = self.OP, self.OP
        self.BLOCKED = None

    def json(self) -> dict: return dict(ST = self.ST, OT = self.OT,
        size = self.size, OP = self.OP, SL = self.SL, TP = self.TP,
        side = self.side.name, mods = self.mods, symbol = self.symbol)

    def margin(self, leverage: int):
        
        bPower = self.symbol.value / self.symbol.point
        return self.size * bPower * self.OP / leverage

    def _modifySL(self, SL: float, actual: float = None) -> None:

        if (SL == None): return object.__setattr__(self, "SL", None)
        self.isPrice(SL)  ;  minDist = self.symbol.minDist
        if (actual == None): actual = self.OP
        limit = actual - self.side.value * minDist
        cond_B = (self.side == Trading.Side.BUY) and (SL <= limit)
        cond_S = (self.side == Trading.Side.SELL) and (SL >= limit)
        if (cond_B or cond_S): return object.__setattr__(self, "SL", self.symbol.round(SL))
        elif not self.strict: return object.__setattr__(self, "SL", self.symbol.round(limit))
        loc, val = ("below", "bid") if (self.side == Trading.Side.BUY) else ("above", "ask")
        raise ValueError(f"SL must be at least {minDist:.5f} {loc} actual {val}.")

    def _modifyTP(self, TP: float, actual: float = None) -> None:

        if (TP == None): return object.__setattr__(self, "TP", None)
        self.isPrice(TP)  ;  minDist = self.symbol.minDist
        if (actual == None): actual = self.OP
        limit = actual + self.side.value * minDist
        cond_B = (self.side == Trading.Side.BUY) and (TP >= limit)
        cond_S = (self.side == Trading.Side.SELL) and (TP <= limit)
        if (cond_B or cond_S): return object.__setattr__(self, "TP", self.symbol.round(TP))
        elif not self.strict: return object.__setattr__(self, "TP", self.symbol.round(limit))
        loc, val = ("above", "bid") if (self.side == Trading.Side.BUY) else ("below", "ask")
        raise ValueError(f"TP must be at least {minDist:.5f} {loc} actual {val}.")

    def checkWorst(self, data: pandas.Series) -> None:

        candle = Candle.From(data)
        low, high = candle.low, candle.high
        WP = self.compare(low, self.WP, high)
        object.__setattr__(self, "WP", WP)

    def checkModify(self, data: pandas.DataFrame) -> None:

        changes = self.modifier.analyze(data, type(self), self.json())
        candle = Candle.From(data)
        actual = self.compare(candle.low, candle.high)
        object.__setattr__(self, "actual", actual)
        if (changes == None): return
        actual = self.compare(candle.bid, candle.ask)
        if ("SL" in changes): self._modifySL(SL = changes["SL"], actual = actual)
        if ("TP" in changes): self._modifyTP(TP = changes["TP"], actual = actual)
        object.__setattr__(self, "mods", self.mods + 1)

    def checkStatus(self, data: pandas.DataFrame) -> Result:

        value, candle = None, Candle.From(data)
        if (value == None): value, result = (self.SL, "SL") if candle.crosses(self.SL) else (None, None)
        if (value == None): value, result = (self.TP, "TP") if candle.crosses(self.TP) else (None, None)
        if (value == None): value, result = self.closer.analyze(data, type(self), self.json())
        if (value == None): return None
        value = self.symbol.round(value)
        return Result(ticket = self.ticket, symbol = self.symbol, side = self.side, size = self.size,
            OT = self.OT, CT = candle.time, ST = self.ST, OP = self.OP, CP = value, mods = self.mods, 
            WP = self.compare(self.WP, value), cause = self.cause, result = result)

#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

class Signal(Trade):

    def __init__(self, ticket: int, symbol: Symbol, side: Trading.Side, ST: datetime, size: float = 1,
        OP: float = None, SL: float = None, TP: float = None, until: datetime = None, cause: str = "",
        strict: bool = False, closer: object = Trading.nullCloser(), modifier: object = Trading.nullModifier()):

        assert isinstance(until, (type(None), datetime)), "Signal should run \"until\" a defined \"datetime\"."
        if isinstance(until, datetime): assert (until > ST), f"Signal should run \"until\" later than runtime."
        super().__init__(ticket, symbol, side, size, 1, ST, ST, 1, None, None, cause, strict, closer, modifier)

        del self.BLOCKED, self.OT, self.WP
        self._modifyOP(OP)
        if self.OP:
            if SL: self._modifySL(SL, actual = self.OP)
            if TP: self._modifyTP(TP, actual = self.OP)
        else:
            if SL: self.isPrice(SL) ; self.SL = SL
            if TP: self.isPrice(TP) ; self.TP = TP
        
        self.until = until
        self.BLOCKED = None
        
    def json(self) -> dict: return dict(ST = self.ST, until = self.until,
        side = self.side.name, OP = self.OP, SL = self.SL, TP = self.TP,
        size = self.size, mods = self.mods, symbol = self.symbol)

    def _modifySize(self, size: float) -> None:
        
        self.size = self._checkSize(size, strict = self.strict)
    
    def _modifyOP(self, OP: float = None) -> None:

        if (OP == None): return object.__setattr__(self, "OP", None)
        self.isPrice(OP)
        object.__setattr__(self, "OP", self.symbol.round(OP))
        if self.SL: self._modifySL(OP - (self.OP - self.SL))
        if self.TP: self._modifyTP(OP - (self.OP - self.TP))

    def checkModify(self, data: pandas.DataFrame) -> None:

        changes = self.modifier.analyze(data, type(self), self.json())
        if (changes == None): return
        candle = Candle.From(data)
        actual = candle.bid if (self.side == self.Side.SELL) else candle.ask
        if ("OP" in changes): self._modifyOP(OP = changes["OP"], actual = actual)
        if ("SL" in changes): self._modifySL(SL = changes["SL"], actual = actual)
        if ("TP" in changes): self._modifyTP(TP = changes["TP"], actual = actual)
        if ("size" in changes): self._modifySize(size = changes["size"])
        object.__setattr__(self, "mods", self.mods + 1)

    def checkExpiry(self, data: pandas.DataFrame) -> None:

        return (self.until != None) and (Candle.From(data).time >= self.until)

    def checkDelete(self, data: pandas.DataFrame) -> None:

        value, result = self.closer.analyze(data, type(self), self.json())
        return (result == "Remove")

    def checkStatus(self, data: pandas.DataFrame) -> Trade:
    
        candle = Candle.From(data)
        actual = candle.bid if (self.side == self.Side.SELL) else candle.ask
        if (self.OP == None): object.__setattr__(self, "OP", actual)
        actual = self.OP - (self.side == self.Side.BUY) * candle.spread
        if candle.crosses(actual): return Trade(
            ticket = self.ticket, symbol = self.symbol, side = self.side, size = self.size,
            ST = self.ST, OT = candle.time, OP = self.OP, SL = self.SL, TP = self.TP, mods = self.mods,
            cause = self.cause, strict = self.strict, closer = self.closer, modifier = self.modifier)
        
#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

class Account:

    def __init__(self, number: int, broker: str = "", initial: float = 10000, marginCall: float = 2):

        self.number, self.broker, self.initial, self.marginCall = number, broker, initial, marginCall

        self._reset()

    def json(self) -> dict:
        
        json = dict(number = self.number, broker = self.broker,
            initial = self.initial, marginCall = self.marginCall)
        if self.trades: json.update(funds = self.funds, equity = self.equity,
            results = self.results, margin = self.margin, level = self.level)
        return json

    def __setattr__(self, name, value):

        if (dir(self)[0] == "BLOCKED"): return
        else: object.__setattr__(self, name, value)

    def _reset(self):

        object.__setattr__(self, "funds", self.initial)
        object.__setattr__(self, "equity", self.initial)
        object.__setattr__(self, "signals", list())
        object.__setattr__(self, "results", list())
        object.__setattr__(self, "trades", list())
        object.__setattr__(self, "margin", 0)
        object.__setattr__(self, "level", 0)

    def _updateEquity(self):

        calc = lambda x: x.sink(Trading.Profit.CURRENCY)
        equity = self.funds - sum(map(calc, self.trades))
        object.__setattr__(self, "equity", equity)
        object.__setattr__(self, "level", equity / self.margin)
        
    def _updateMargin(self, trade: Trade):

        calc = lambda x: x.margin(self.leverage)
        if (trade != None): margin = self.margin + calc(trade)
        else: margin = sum(map(calc, self.trades))
        object.__setattr__(self, "margin", margin)

    def _updateFunds(self, result: Result):

        calc = lambda x: x.gain(Trading.Profit.CURRENCY)
        if (result != None): funds = self.funds + calc(result)
        else: funds = self.initial + sum(map(calc, self.results))
        object.__setattr__(self, "funds", funds)