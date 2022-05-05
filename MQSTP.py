import os, numpy, json as JSON
from pandas import Series, DataFrame
from pandas import Timedelta, Timestamp
from pandas import to_datetime as datetime
from pandas import read_html, read_csv, concat
from matplotlib.figure import Figure, Axes
from matplotlib.pyplot import style as PlotStyle
from IPython.display import display as print
from scipy import stats
from itertools import product
from ProgBar import ProgBar

#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

def specs(quotes: list, csv: str = "") -> DataFrame:
    if not csv: return Series(dict.fromkeys(quotes))
    return read_csv(csv, index_col = 0).loc[quotes, :]

#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

class TickRange:

    __defaultLinBases, __defaultLogBases = numpy.array([1, 2, 4, 5, 10]), numpy.array([1, 2, 5])

    @classmethod
    def lin(cls, xMin: float, xMax: float, maxLength: int = 50) -> numpy.ndarray:
        """
        Get a range of linearly spaced values based on two end inputs. Space (tick) will be smaller
        or equal to the distance between both inputs, divided by "`maxLength`". Then, resulting extreme
        values will be a result of "`xMin`" and "`xMax`" rounded down and up respectively towards the
        calculated tick. The significand of such tick will be a common divisor of powers of 10 =>
        `[1, 2, 4, 5, 10]`.\n
        The resulting array:\n
        -> ...will never be larger than "`maxLength`".\n
        -> ...will always include zero if it has both positive and negative values.\n
        Some examples for different (`xMin, xMax, maxLength`):\n
        -> ( `1.0`, `5.0`, `10`) => tick = 0.5 => `[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]`\n
        -> ( `1.0`, `5.0`, `11`) => tick = 0.4 => `[1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8, 4.2, 4.6, 5.0]`\n
        -> (`-1.0`, `5.0`, `11`) => tick = 1.0 => `[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]`\n
        -> (`-1.0`, `2.0`, `10`) => tick = 0.4 => `[-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6, 2.0]`\n
        -> (`-100`, `200`, `10`) => tick = 40 => `[-120, -80, -40, 0, 40, 80, 120, 160, 200]`\n
        -> (`-100`, `201`, `10`) => tick = 40 => `[-120, -80, -40, 0, 40, 80, 120, 160, 200, 240]`\n
        -> (`-100`, `100`, `20`) => tick = 20 => `[-100, -80, -40, -20, 0, 20, 40, 60, 80, 100]`\n
        -> (`-100`, `100`, `21`) => tick = 10 => `[-100, -90, -80..., -20, -10, 0, 10, 20, ...80, 90, 100]`\n
        Serves as a linear grid-generating method. Values can be any real number.
        """
        tick = abs(xMax - xMin) / maxLength
        ints = numpy.array(cls.__defaultLinBases)
        exp10 = 10.0 ** numpy.floor(numpy.log10(tick))
        tick = ints[tick / exp10 < ints][0] * exp10
        xMin = int(numpy.floor(xMin / tick)) * tick
        xMax = int(numpy.ceil(xMax / tick)) * tick
        n = int((xMax - xMin) / tick) + 1
        return numpy.linspace(xMin, xMax, n)

    @classmethod
    def log(cls, xMin: float, xMax: float, ints: list = None) -> numpy.ndarray:
        """
        Get a range of logarithmically spaced value intervals based on two end inputs. "`ints`" is a base
        list for the int coefficients of such spaces. Default is => `[1, 2, 5]`. Range will go from the
        previous "1 x 10^nMin" of "`xMax`", to the next "int x 10^nMax" of "`xMax`", Each "n" being the
        order of magnitude of its number.\n
        Some examples for different (`xMin, xMax, ints`):\n
        -> (`123`, `4567`, `[1, 2, 5]`) => nMin = 2, nMax = 3 => `[100, 200, 500, 1000, 2000, 5000]`\n
        -> (`123`, `5678`, `[1, 2, 5]`) => nMin = 2, nMax = 4 => `[100, 200, 500, 1000, 2000, 5000, 10000]`\n
        -> (`123`, `2000`, `[1, 2, 5]`) => nMin = 2, nMax = 3 => `[100, 200, 400, 1000, 2000, 5000]`\n
        -> (`123`, `2000`, `[1, 2, 4]`) => nMin = 2, nMax = 3 => `[100, 200, 400, 1000, 2000, 4000]`\n
        -> (`123`, `1999`, `[1, 2, 4]`) => nMin = 2, nMax = 3 => `[100, 200, 400, 1000, 2000]`\n
        -> ( `99`, `5678`, `[1, 2, 5]`) => nMin = 1, nMax = 3 => `[10, 20, 50, 100..., 2000, 5000, 10000]`\n
        Serves as a logarithmic grid-generating method. NOTE: Values CANNOT be negative.
        """
        if isinstance(ints, type(None)):
            ints = cls.__defaultLogBases
        ticks = numpy.log10([xMin, xMax])
        ticks = numpy.arange(*(ticks + [0, 2]))
        ticks = numpy.power(10, ticks.astype(int))
        ticks = numpy.outer(ticks, ints).reshape(-1)
        return ticks[: sum(ticks <= xMax) + 1]

#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

class Parse:

    #===#===#===#===#===#===##===#===#===#===#===#===##===#===#===#===#===#===#===#===#===#===#===#===#===#

    @classmethod
    def fromBacktestMQL4(cls, htmlFile: str, specFile: str, toJSON = False) -> dict:

        struct = DataFrame(index = ["index", "type"], data = {
         "time": [1, str], "side": [2, str], "#": [3, int],
         "size": [4, float], "price": [5, float], }, )
         
        with open(htmlFile, "r", errors = "ignore") as file:
            file = file.read()
            meta = file[file.find("<div") : file.find("<table")]
            tag_1 = "<div style=\"font: 16pt Times New Roman\"><b>"
            tag_2 = "<div style=\"font: 10pt Times New Roman\"><b>"
            meta = meta[meta.find(tag_1) + len(tag_1) :]
            expert = meta[: meta.find("<")]
            meta = meta[meta.find(tag_2) + len(tag_2) :]
            broker = meta[: meta.find("-")]
            header, orders = read_html(file)[: 2]
            
        quote, timeframe, config_, config = *header.iloc[[0, 1, 3], 2], dict()
        timeframe = timeframe[1 + timeframe.index("(") : timeframe.index(")")]
        config_ = config_.replace("false", "False").replace("true", "True")
        for line in config_[: -1].split("; "):
            key, value = line.split("=")
            try: value = eval(value)
            finally: config[key] = value
        quote = quote[: quote.find(" ")]
        
        columns = orders.columns[struct.loc["index"].to_list()]
        rename = dict(zip(columns, struct.columns))
        orders.rename(inplace = True, columns = rename)
        orders = orders.loc[1 :, struct.columns]
        orders = orders.astype(struct.loc["type"])
        orders.set_index(inplace = True, keys = "#")
        orders["time"] = datetime(orders["time"])
        orders["time"] = orders["time"].astype(str)
        
        columns = ["ST", "OT", "CT", "OP", "CP", "WP",
            "quote", "side", "size", "mods", "cause"]
        trades = DataFrame(columns = columns)
        for trade in orders.index:
            order = orders.loc[trade, :]
            time, side, size, price = order.values.T
            exec = (side == "buy") | (side == "sell")
            exec = order.loc[exec, :]
            trades.at[trade, "ST"] = time[0]
            trades.at[trade, "CT"] = time[-1]
            trades.at[trade, "CP"] = price[-1]
            trades.at[trade, "quote"] = quote
            trades.at[trade, "size"] = size[0]
            trades.at[trade, "cause"] = side[-1]
            trades.at[trade, "mods"] = len(time)
            time, side, size, OP = exec.values[0]
            fside = min if (side == "buy") else max
            trades.at[trade, "WP"] = fside(price)
            trades.at[trade, "side"] = side
            trades.at[trade, "OT"] = time
            trades.at[trade, "OP"] = OP

        obj = {"expert": expert, "broker": broker,
            "timeframe": timeframe, "config": config,
            "quotes": specs([quote], specFile), "trades": trades}
        
        if not toJSON: return obj
        dot = htmlFile[: : -1].find(".") + 1
        fileName = htmlFile[: -dot] + ".json"
        with open(fileName, "w") as file:
            obj2 = {"source": "backtrack"}  ;  obj2.update(obj)
            obj2["quotes"] = obj["quotes"].transpose().to_dict()
            obj2["trades"] = {
                "columns": ["#"] + list(trades.columns),
                "data": trades.reset_index().values.tolist() }
            JSON.dump(indent = 4, fp = file, obj = obj2)
        return obj

    #===#===#===#===#===#===##===#===#===#===#===#===##===#===#===#===#===#===#===#===#===#===#===#===#===#

    @classmethod
    def fromHistoryMQL4(cls, htmlFile: str, specFile: str, toJSON = False) -> dict:

        struct = DataFrame(data = {
            "#": [0, int], "OT": [1, str], "CT": [8, str],
            "OP": [5, float], "CP": [9, float], "SL": [6, float],
            "TP": [7, float], "quote": [4, str], "side": [2, str],
            "size": [3, float], }, index = ["index", "type"] )
            
        with open(htmlFile, "r", errors = "ignore") as file:
            file = file.read()
            meta = file[file.find("<div") : file.find("<table")]
            tag_1 = "<div style=\"font: 20pt Times New Roman\"><b>"
            meta = meta[meta.find(tag_1) + len(tag_1) :]
            broker = meta[: meta.find("<")]
            trades = read_html(file)[0]
            
        header, trades = trades.iloc[:2], trades.iloc[3:]
        account = int(header.loc[0, 0].split(": ")[1])
        leverage = int(header.loc[0, 9].split(":")[2])
        
        nan = numpy.where(trades.iloc[:, 9].isna())[0][0]
        trades = trades.iloc[: nan, :]
        comms = numpy.where(trades.loc[:, 13].isna())
        comms = trades.iloc[comms[0], 10].to_list()
        rename = dict(zip(struct.loc["index"], struct.columns))
        trades = trades.iloc[: nan, list(rename.keys())]
        trades.rename(inplace = True, columns = rename)
        trades["side"] = trades["side"].str.capitalize()
        trades["quote"] = trades["quote"].str.upper()
        exec = (trades["side"] == "Buy") | (trades["side"] == "Sell")
        trades = trades.loc[exec, :]
        trades["OT"] = datetime(trades["OT"])
        trades["CT"] = datetime(trades["CT"])
        trades = trades.astype(struct.loc["type"])
        trades.set_index(inplace = True, keys = "#")
        SL = (abs(trades["CP"] / trades["SL"] - 1) <= 1e-5)
        TP = (abs(trades["CP"] / trades["TP"] - 1) <= 1e-5)
        trades[["cause", "comment"]] = ""
        trades.loc[SL, "cause"], trades.loc[TP, "cause"] = "SL", "TP"
        trades.drop(inplace = True, columns = ["SL", "TP"])
        try: trades["comment"] = comms
        except: pass

        obj = {"account": account, "broker": broker, "leverage": leverage,
            "quotes": specs(set(trades["quote"]), specFile), "trades": trades}

        if not toJSON: return obj
        dot = htmlFile[: : -1].find(".") + 1
        fileName = htmlFile[: -dot] + ".json"
        with open(fileName, "w") as file:
            obj2 = {"source": "history"}  ;  obj2.update(obj)
            obj2["quotes"] = obj["quotes"].transpose().to_dict()
            obj2["trades"] = {
                "columns": ["#"] + list(trades.columns),
                "data": trades.reset_index().values.tolist() }
            JSON.dump(indent = 4, fp = file, obj = obj2)
        return obj

#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

class Record:
    
    _quoteColumns = ["point", "value"]
    _tradeColumns = ["#", "OT", "CT", "OP", "CP", "quote", "side", "size"]
    _backColumns = ["Balance", "Equity", "Margin", "Level", "Return", "Drawdown"]
    _foreColumns = ["min", "mid", "max"]

    _plotColors = ["tomato", "limegreen", "turquoise", "magenta", "gold", "skyblue", "hotpink", 
                   "orange", "indigo", "yellowgreen", "orangered", "olive", "fuchsia", "khaki"]

    _defStyle = "https://raw.githubusercontent.com/gsolaril/"\
        + "Templates4Coding/master/Python/mplfinance.mplstyle"

    #===#===#===#===#===#===##===#===#===#===#===#===##===#===#===#===#===#===#===#===#===#===#===#===#===#

    @classmethod
    def fromJson(cls, path: str, **kwargs):

        with open(path, "r") as file: json = JSON.load(file)
        quotes = DataFrame(json["quotes"])
        columns = json["trades"]["columns"]
        trades = json["trades"]["data"]
        trades = DataFrame(trades, columns = columns)
        metadata = json.copy()
        metadata.pop("trades")
        metadata.pop("quotes")
        return cls(trades, quotes,
            leverage = json["leverage"],
            metadata = Series(metadata),
            **kwargs)

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def __init__(self, trades: DataFrame, quotes: DataFrame,
                 leverage: float, metadata: dict = dict()):

        _tColumns = set(self._tradeColumns)
        _qColumns = set(self._quoteColumns)
        assert isinstance(trades, DataFrame) and len(trades), "\"trades\" must be a Pandas' DataFrame."
        assert isinstance(quotes, DataFrame) and len(quotes), "\"quotes\" must be a Pandas' DataFrame."
        assert isinstance(leverage, (int, float)) and (leverage >= 1), "\"leverage\" must be a number > 1."
        assert set(trades.columns).issuperset(_tColumns), "\"trades'\" columns must include: %s" % _tColumns
        assert set(quotes.index).issuperset(_qColumns), "\"quotes'\" row names must include: %s" % _qColumns
        assert isinstance(metadata, (dict, Series)), "\"metadata\" must be a dict or Pandas' Series."
        quotes_missing = set(trades["quote"]).difference(set(quotes.columns))
        assert not quotes_missing, "\"quotes\" missing: %s" % quotes_missing

        self.trades = trades.copy()
        self.quotes = quotes.copy().astype(float)
        self.lever = leverage
        self.trades.set_index("#", inplace = True)
        self.trades = self.trades.infer_objects()
        self.trades["OT"] = datetime(self.trades["OT"])
        self.trades["CT"] = datetime(self.trades["CT"])
        self.metadata = metadata.copy()
        self.metaplot = self.__metaplot()
        if not ("WP" in self.trades.columns):
            self.trades.loc[:, "WP"] = numpy.nan

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def __metaplot(self):

        source = self.metadata["source"]
        t1 = self.trades["OT"].iat[0].strftime("%Y/%m/%d")
        t2 = self.trades["CT"].iat[-1].strftime("%Y/%m/%d");  config = [ list() ]
        broker = "" if not ("broker" in self.metadata) else self.metadata["broker"]
        expert = "" if not ("expert" in self.metadata) else self.metadata["expert"]
        account = "" if not ("account" in self.metadata) else self.metadata["account"]
        tf = "" if not ("timeframe" in self.metadata) else self.metadata["timeframe"]
        footnote = list()

        if (source == "history"):
            self.title = "history of %d @ %s" % (account, broker)
            self.title += ": %s - %s (1:%s)" % (t1, t2, self.lever)
        if (source == "backtrack"):
            self.title = "backtrack of %s @ %s" % (expert, broker)
            self.title += ": %s - %s @ %s (1:%s)" % (t1, t2, tf, self.lever)
        if ("config" in self.metadata):
            line = "" ; separator = ",   "
            for key, value in self.metadata["config"].items():
                if isinstance(value, str): value = f"\"{value}\""
                entry = "\"%s\" = %s" % (key, value) + separator
                if not (len(line) + entry > 150): line += entry
                else: footnote.append(line.rstrip()) ; line = ""
        
        self.footnote = "\n".join(footnote)

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    @staticmethod
    def __save(figure: Figure, title: str):

        nFile = 0
        title = title.replace("/", ".")
        title = title.replace("\n", " ")
        title = title.replace("1:", "L")
        title = title.replace(":", ",")
        filename = "./" + title + ".jpg"
        while os.path.exists(filename):
            filename = "./" + title
            filename += " (%d).jpg" % nFile
            nFile += 1
        figure.savefig(filename)

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#
        
    def seekWP(self, database: DataFrame, strict: bool = False) -> None:

        try: columns = database.columns
        except: raise AssertionError("\"database\" must be a DataFrame")
        ab = {"ask", "bid"} ; need = ["OT", "CT", "OP", "CP", "side"]
        try: bothSides = ab in map(set, columns.levels)
        except: bothSides = ab.issubset(columns)
        print("\nCompleting sinks...")
        progBar = ProgBar(
            items = self.trades.shape[0],
            width = 40, verbose = "Trade")

        for n in self.trades.index:

            trade = self.trades.loc[n, need]
            OT, CT, OP, CP, side = trade.values
            isSell = (side[0].upper() == "S")
            closer = ["ask" if isSell else "bid"]
            if not bothSides: closer = columns
            prices = numpy.array([OP, CP])
            period = (OT <= database.index) & (database.index <= CT)
            if any(period): prices = database[closer][period].values
            prices = prices.reshape(-1)
            minPrice, maxPrice = prices.min(), prices.max()
            below_OT, above_OT = (minPrice <= OT), (OT <= maxPrice)
            below_CT, above_CT = (minPrice <= CT), (CT <= maxPrice)
            if (not strict) or (below_OT & above_OT & below_CT & above_CT):
                self.trades.loc[n, "WP"] = maxPrice if isSell else minPrice
            progBar.show()

        sign = lambda x: -1 if (x.lower() == "sell") else 1
        self.trades["sink"] = self.trades["WP"] - self.trades["OP"]
        if not strict: self.trades["sink"] = - self.trades["sink"].abs()
        else: self.trades["sink"] *= self.trades["side"].apply(sign)
        point = self.quotes.loc["point", self.trades["quote"]]
        self.trades["sink"] = self.trades["sink"] / point.values
        self.trades["sink"] = self.trades["sink"].astype(int)

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def distribute(self, **kwargs) -> None:

        sign = lambda x: -1 if (x.lower() == "sell") else 1
        point = self.quotes.loc["point", self.trades["quote"]]
        self.trades["gain"] = self.trades["side"].apply(sign)
        self.trades["gain"] *= self.trades["CP"] - self.trades["OP"]
        self.trades["gain"] = self.trades["gain"] / point.values
        self.trades["gain"] = self.trades["gain"].astype(int)
        self.trades["sink"] = self.trades["side"].apply(sign)
        self.trades["sink"] *= self.trades["WP"] - self.trades["OP"]
        self.trades["sink"] = self.trades["sink"] / point.values
        try: self.trades["sink"] = self.trades["sink"].astype(int)
        except: pass
        isSink = (self.trades["sink"].count() != 0)

        with PlotStyle.context(
            kwargs.get("style", self._defStyle)):
                self.distribution = self.__plotDist(**kwargs)


    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#
    
    def __plotDist(self, **kwargs) -> dict[str, dict]:
        
        quotes = self.trades["quote"]
        pv = self.quotes.loc["value", quotes.values]
        pv.index = quotes.index; quotes = set(quotes)

        Fig = Figure()
        save = kwargs.pop("save", False)
        rows = len(quotes) + (len(quotes) >= 2)
        width, height = kwargs.pop("figsize", (15, 5))
        width, height = width, height * rows
        Fig.set_figheight(height); Fig.set_figwidth(width)

        fSize = kwargs.pop("fontsize", 12)
        title = "Histogram & QQ plot - " + self.title
        underlined = title + "\n" + "‾‾‾" * len(title)
        Fig.suptitle(underlined, fontsize = int(fSize * 1.1))
        Fig.supxlabel(self.footnote, fontsize = fSize)

        axes = dict.fromkeys(sorted(quotes))

        print("\nCreating statistical plots...")
        labelArgs = {"fontsize": int(fSize * 1.1), "va": "bottom"}
        items = list(product(axes.keys(), ["Histogram", "QQ"]))
        items = list(map(lambda item: ", ".join(item), items))
        progBar = ProgBar(items, width = 40, verbose = "Plotting")
        nPts = int(width * 80/3)

        for n, row in enumerate(axes.keys()):

            quote = row
            left, right = (2 * n + 1, 2 * n + 2)
            selected = (self.trades["quote"] == quote)
            gain = self.trades.loc[selected, "gain"]
            ax_L = Fig.add_subplot(len(axes), 2, left) 
            ax_R = Fig.add_subplot(len(axes), 2, right)
            plotArgs = {"values": gain, "fSize": fSize}
            self.__histPlot(legend = row, nPts = nPts,
                ax = ax_L, **plotArgs) ; progBar.show()
            self.__qqPlot(
                ax = ax_R, **plotArgs) ; progBar.show()
            if (list(axes.keys())[-1] == row):
                ax_L.xaxis.set_label_position("top")
                ax_R.xaxis.set_label_position("top")
                ax_L.set_xlabel("Gain (points)", **labelArgs)
                ax_R.set_xlabel("Std. deviations", **labelArgs)
                ax_R.legend(loc = "lower right", fontsize = fSize)
            ax_L.minorticks_off(); ax_R.minorticks_off()
            axes[row] = {"histogram": ax_L, "QQ": ax_R}

        Fig.set_tight_layout((0, 0, 1, 1))
        print("\nHistograms & QQ plots finished!")
        if save:
            print("\nSaving histograms & QQ plots...")
            self.__save(Fig, title)
            print("...Histograms & QQ plots saved!")
        return axes

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def __histPlot(self, ax: Axes, values: Series, fSize: float, nPts: int, legend: str):

        vLineArgs = {"marker": "s", "ms": 5, "lw": 1.5}
        labelArgs = {"fontsize": int(fSize * 1.1), "va": "bottom"}
        histArgs = {"label": legend, "color": "gold", "alpha": 1/2}

        xMax = values.abs().max()
        xSims = TickRange.lin(-xMax, +xMax, nPts)
        xBars = TickRange.lin(-xMax, +xMax, nPts // 4)
        xTicks = TickRange.lin(-xMax, +xMax, nPts // 8)
        xTicks = xTicks[xTicks != 0].astype(values.dtype)
        barWidth = xBars[1] - xBars[0]
        hist = values.value_counts(bins = xBars)
        xBars, yBars = hist.index.mid, hist.values
        ax.bar(x = xBars, height = yBars, width = barWidth, **histArgs)

        ax.set_xticks(xTicks)
        if (2 * barWidth <= 1e0): xTicks = [f"%.2f" % x for x in xTicks]
        if (2 * barWidth >= 1e3): xTicks = ["%dK" % (x / 1e3) for x in xTicks]
        if (2 * barWidth >= 1e6): xTicks = ["%dM" % (x / 1e6) for x in xTicks]
        ax.set_xticklabels(xTicks, fontsize = fSize, rotation = 90)
        
        ax.set_yscale("log")  ;  yMode = yBars.max()

        marg = (yMode ** 0.05) * 1.414
        xMean, xSTDV = stats.norm.fit(values)
        xM2SD = xMean - 2 * xSTDV
        yMean = len(values) * stats.norm.pdf(0, 0, xSTDV)
        yM2SD = len(values) * stats.norm.pdf(xM2SD, xMean, xSTDV) * yMode / yMean
        ySims = len(values) * stats.norm.pdf(xSims, xMean, xSTDV) * yMode / yMean
        yMin, yMax, yMean = 1 / marg, (marg * yMode), yMean * yMode / yMean
        yMean, yM2SD = max(1, yMean), max(1, yM2SD)

        ax.axvline(x = 0, color = "silver", lw = 1, ls = "--")

        yTicks = TickRange.log(1, yMax)
        ax.set_yticks(yTicks) ; ax.set_yticklabels(yTicks)
        ax.plot(xSims, ySims, color = "dodgerblue", lw = 2)
        sMean, sM2SD = "  $\\mu$", "$\\mu-2\\sigma$  "
        ax.text(xMean, yMean, sMean, **labelArgs, ha = "left")
        ax.text(xM2SD, yM2SD, sM2SD, **labelArgs, ha = "right")
        ax.plot(2 * [xMean], [0, yMean], **vLineArgs, color = "limegreen")
        ax.plot(2 * [xM2SD], [0, yM2SD], **vLineArgs, color = "orangered")
        if (yMode == 1): ax.axes.get_yaxis().set_visible(False)

        ax.legend(fontsize = int(fSize * 1.1), loc = "upper left")

        ax.grid(True) ; ax.set_xlim(-xMax, xMax) ; ax.set_ylim(yMin, yMax)

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def __qqPlot(self, ax: Axes, values: Series, fSize: float):

        yMean_1, aSTDV_1 = stats.norm.fit(values)
        qqArgs = {"x": values, "fit": True}
        (xPoints, yPoints), (yMean_2, aSTDV_2, R) = stats.probplot(**qqArgs)
        Stats = {"$\mu$": yMean_1, "$\sigma$": aSTDV_1, "$S$": yMean_1 / aSTDV_1, "$R^2$": R**2}
        ax.scatter(xPoints, yPoints, s = 16, marker = "o", c = "hotpink", label = "Samples")

        yMin, yMax = values.min(), values.max()
        yLim = max(abs(yMin), abs(yMax))
        xLim = (yLim - yMean_1) / aSTDV_1

        xy3 = lambda ym, sd: ([-xLim, xLim], [ym - xLim * sd, ym + xLim * sd])

        lineArgs = {"lw": 3, "color": "lime", "alpha": 3/4}
        ax.plot(*xy3(yMean_1, aSTDV_1), **lineArgs, label = "PDF", ls = "-")
        ax.plot(*xy3(yMean_2, aSTDV_2), **lineArgs, label = "Fit", ls = "--")

        xTicks = numpy.arange(-3, 4)  ;  ax.set_xticks(xTicks)
        xTicks = map(lambda x: "$\mu$" * (x == 0), xTicks)
        ax.set_xticklabels(list(xTicks), fontsize = int(fSize * 1.2))
        ax.plot([0, 0], [-yLim, yMean_1], "s:w", lw = 3, alpha = 2/3)
        ax.plot([-xLim, 0], 2 * [yMean_1], "s:w", lw = 3, alpha = 2/3)
        legend = "\n".join(["%s = %.4g" % (K, V) for K, V in Stats.items()])
        textArgs = {"fontsize": int(fSize * 1.2), "ha": "left", "va": "top"}
        ax.text(0.05, 0.95, legend, transform = ax.transAxes, **textArgs)
        ax.set_xlim(-xLim, xLim)  ;  ax.set_ylim(-yLim, yLim)

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def backtrack(self, initFunds: float = 10000,
        initSizer: int = 1, useSizes: bool = False,
        sizeByEquity: bool = False, **kwargs) -> None:
        
        funds, sizes = initFunds, initSizer
        if useSizes: sizes *= self.trades["size"]
        timeline = self.trades[["OT", "CT"]].melt()["value"]
        timeline = timeline.sort_values().drop_duplicates()
        print("\nProcessing trades...")
        self.backtrack_f0 = self.__backtrack(funds, sizes, timeline)
        sizeBy = "Equity" if sizeByEquity else "Balance"
        self.growth = self.backtrack_f0.loc[self.trades["OT"], [sizeBy]]
        self.growth = self.growth.set_index(self.trades.index)[sizeBy]
        if useSizes: self.growth *= self.trades["size"]
        self.growth = (self.growth / initFunds).round(2)
        print("\nCompounding trades...")
        self.backtrack_f1 = self.__backtrack(funds, self.growth, timeline)
        with PlotStyle.context(
            kwargs.get("style", self._defStyle)):
                self.backplot = self.__backPlot(**kwargs)

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def __backtrack(self, _funds: float, _sizes: numpy.array, _timeline: numpy.array) -> DataFrame:

        print("\nCreating backtrack...")
        backtrack = {"columns": self._backColumns, "index": _timeline}
        backtrack = DataFrame(data = 0, dtype = float, **backtrack)
        _sizes *= Series(data = 1, index = self.trades.index)
        needed = ["OT", "CT", "OP", "gain", "sink", "quote"]
        
        progBar = ProgBar(
            width = 40, verbose = "Trade",
            items = self.trades.shape[0])
        for ID in self.trades.index:
            size, trade = _sizes.at[ID], self.trades.loc[ID]
            ot, ct, op, gain, sink, quote = trade.loc[needed]
            pv, ps = self.quotes[quote][["value", "point"]]
            backtrack.at[ct, "Balance"] += gain * size * pv
            backtrack.at[ot, "Equity"] -= sink * size * pv
            margin = size * op * (pv / ps) / self.lever
            backtrack.at[ot, "Margin"] += margin
            backtrack.at[ct, "Margin"] -= margin
            progBar.show()
            
        backtrack["Balance"] = backtrack["Balance"].cumsum() + _funds
        backtrack["Equity"] = backtrack["Equity"].fillna(0)
        backtrack["Equity"] += backtrack["Balance"]
        backtrack["Margin"] = backtrack["Margin"].cumsum()
        backtrack["Level"] = backtrack["Equity"] / backtrack["Margin"]
        backtrack["Return"] = backtrack["Balance"].pct_change()
        backtrack["Return"] = backtrack["Return"].fillna(0) + 1
        backtrack["Return"] = numpy.log(backtrack["Return"])
        drawUp = backtrack["Return"].cumsum()
        drawUp = drawUp / drawUp.cummax()
        backtrack["Drawdown"] = 1 - drawUp.fillna(1)
        return backtrack      

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def __backPlot(self, **kwargs) -> dict[str, Axes]:
        
        print("\nDrawing backtrack...")
        progBar = ProgBar(width = 40, verbose = "Drawing",
            items = ["figure & axes", "balance & equity", "margin",
                "weakpoints", "x-grid", "activity plot", "y-grid"])

        Fig = Figure()
        save = kwargs.pop("save", False)
        width, height = kwargs.get("figsize", (15, 10))
        Fig.set_figheight(height); Fig.set_figwidth(width)

        fSize = kwargs.pop("fontsize", 12)
        title = "Backtrack - " + self.title
        underlined = title + "\n" + "‾‾‾" * len(title)
        Fig.suptitle(underlined, fontsize = int(fSize * 1.1))
        Fig.supxlabel(self.footnote, fontsize = fSize)

        Ax2 = Fig.add_subplot(5, 1, (5, 5))
        Ax1 = Fig.add_subplot(5, 1, (1, 4), sharex = Ax2)
        Fig.set_tight_layout((0, 0.15, 1, 0.70))
        progBar.show()

        yMaxS = self.backtrack_f0.iloc[:, : 3].max().max()
        yMaxC = self.backtrack_f1.iloc[:, : 3].max().max()
        yMin = self.backtrack_f0["Margin"].mean()
        yMax = max(yMaxS, yMaxC)
        yTicks = TickRange.log(yMin, yMax)
        
        timeline = self.backtrack_f0.index
        s = "Equity" ; Ax1.fill_between(x = timeline, color = "dodgerblue", lw = 2,
            alpha = 0.5, y1 = self.backtrack_f0[s], y2 = self.backtrack_f1[s], label = s)
        s = "Balance" ; Ax1.fill_between(x = timeline, color = "limegreen", lw = 2,
            alpha = 0.75, y1 = self.backtrack_f0[s], y2 = self.backtrack_f1[s], label = s)
        Ax1.set_yscale("log")
        progBar.show()

        Ax1.set_yticks(yTicks) ; Ax1.grid(True, lw = 3, alpha = 1/5)
        expFormat = lambda x: r"$%s \times 10^%d$" % (str(x)[:1], numpy.log10(x))
        Ax1.set_yticklabels(map(expFormat, yTicks), fontsize = fSize)
        margin_f0 = self.backtrack_f0["Margin"].copy().resample("D").max()
        margin_f1 = self.backtrack_f1["Margin"].copy().resample("D").max()
        Ax1.bar(x = margin_f0.index, bottom = margin_f0.values, width = 5, color = "yellow", 
                height = margin_f1.values - margin_f0.values, alpha = 1/3, label = "Margin")
        progBar.show()
        
        m, s = stats.norm.fit(self.trades["gain"])
        VARs = (self.trades["gain"] <= m - 2 * s)
        VARs = self.trades.loc[VARs, "CT"]
        VARs = self.backtrack_f0.loc[VARs.values, "Balance"]
        tVAR, pVAR = VARs.index, VARs.values
        scatterArgs = {"lw": 3, "ec": "orangered", "fc": "none", "marker": "o"}
        Ax1.scatter(tVAR, pVAR, **scatterArgs, s = 40 ** 2)
        Ax1.scatter(0, 0, **scatterArgs, label = r"$\mu - 2\sigma$")  
        progBar.show()

        tDelta = timeline[-1] - timeline[0]
        tSteps = numpy.arange(61) * tDelta / 60
        xTicks = datetime(timeline[0] + tSteps)
        Ax2.set_xticks(xTicks)
        fmFreq = fmTick = lambda x: x.strftime("%Y/%m/%d")
        period, delta = 30, Timedelta(1, "day")
        if (tDelta.days < 60):
            period, delta = 24, Timedelta(1, "hour")
            fmFreq = lambda x: x.strftime("%Y/%m/%d %H:00")
            fmTick = lambda x: x.strftime("%m/%d %Hh")
        if (tDelta.days < 10):
            period, delta = 60, Timedelta(1, "minute")
            fmFreq = lambda x: x.strftime("%Y/%m/%d %H:%M")
            fmTick = lambda x: x.strftime("%d, %H:%M")
        xTickLabels = list(map(fmTick, xTicks))
        Ax2.set_xticklabels(xTickLabels, fontsize = fSize, rotation = 90)
        Ax1.set_xticklabels(xTickLabels, fontsize = fSize, color = "none")
        Ax2.grid(True, lw = 3, alpha = 1/5)
        progBar.show()

        freq = self.trades["OT"].apply(fmFreq)
        freq = freq.value_counts().sort_index()
        freq.index = datetime(freq.index)
        tDelta = freq.index[-1] - freq.index[0]
        tSteps = numpy.arange(tDelta / delta + 1)
        timeline = freq.index[0] + tSteps * delta
        freq_filled = Series(0, index = timeline)
        progBar.show()
        
        freq_filled.loc[freq.index] = freq.values
        freq_filled = freq_filled.sort_index()
        Ax2.bar(x = freq_filled.index, height = freq_filled,
                color = "purple", width = 1, alpha = 1)
        freq_SMA = freq_filled.rolling(window = period).mean()
        Ax2.plot(freq_SMA.index, freq_SMA.values, lw = 3,
            color = "orange", label = "SMA(%d)" % period)
        yTicks2 = range(freq_filled.max() + 1)

        Ax2.set_yticks(yTicks2)
        Ax2.set_ylim(0, yTicks2[-1])
        Ax2.set_yticklabels(yTicks2, fontsize = fSize)
        Ax1.set_xlim(timeline[0], timeline[-1])
        Ax1.set_ylim(yMin, yMax)
        progBar.show()
        
        Ax1.set_ylabel(ylabel = "USD")
        Ax2.set_ylabel(ylabel = "# trades")
        Ax1.legend(fontsize = int(fSize * 1.1), loc = "upper left")
        Ax2.legend(fontsize = int(fSize * 1.1))
        print("\nBacktrack finished!")
        if save:
            print("\nSaving backtrack...")
            self.__save(Fig, title)
            print("...Backtrack saved!")
        return {"balance": Ax1, "frequency": Ax2}

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def foretrack(self, until: Timestamp, start: Timestamp = None, sizeByEquity: bool = False,
    iterations: int = 100, initFunds: float = 10000, initSize: int = 1, outPropFreq: float = 2,
    **kwargs) -> None:

        print("\nCreating foretrack...")
        if (start == "last"): start = self.trades["CT"].iat[-1]
        if not isinstance(start, Timestamp): start = Timestamp("now")
        if not isinstance(until, Timestamp): until = Timestamp(until)
        tMean = self.trades["CT"].diff().mean()
        assert (until - start > tMean), "Not enough time steps!"
        timeline = numpy.arange(start, until, tMean)
        self.foretrack_f0 = DataFrame(columns = self._foreColumns, index = timeline)

        pv = self.quotes.loc["value"]
        quotes = self.trades["quote"].copy()
        qProb = quotes.value_counts() / len(quotes)
        gMean, gSTDV = qProb.copy(), qProb.copy()
        sMean, sSTDV = qProb.copy(), qProb.copy()
        quotes = qProb.index

        for quote in quotes:
            qTrades = self.trades.loc[self.trades["quote"] == quote]
            gMean.at[quote], gSTDV.at[quote] = stats.norm.fit(qTrades["gain"].fillna(0))
            sMean.at[quote], sSTDV.at[quote] = stats.norm.fit(qTrades["sink"].fillna(0))

        gMin = self.trades["gain"].min()
        gM2SD = (gMean - 2 * gSTDV).min()
        outliers = (self.trades["gain"] < gM2SD)
        oProb = self.trades.loc[outliers].shape[0]
        oProb *= outPropFreq / self.trades.shape[0]
        oProb = [1 - oProb, oProb]

        progBar = ProgBar(items = iterations, width = 40, verbose = "Iteration")
        while iterations:
            iterations -= 1  ;  progBar.show()
            rndQuotes = numpy.random.choice(quotes, timeline.size, p = qProb)
            outliers = numpy.random.choice([1, 0], timeline.size, p = oProb)
            _pv = pv.loc[rndQuotes].values
            _gMean = gMean.loc[rndQuotes].values
            _gSTDV = gSTDV.loc[rndQuotes].values
            _sMean = sMean.loc[rndQuotes].values
            _sSTDV = sSTDV.loc[rndQuotes].values
            rWalkMax = numpy.random.randn(timeline.size) * _gSTDV + _gMean
            rWalkMin = rWalkMax - sizeByEquity * (abs(_sMean) - 2 * _sSTDV)
            rWalkMax[outliers] = gMin   ;   rWalkMin[outliers] = gMin
            rWalkMax[0], rWalkMin[0] = 0, 0
            rWalkMax = Series(rWalkMax, index = timeline).cumsum()
            rWalkMin = Series(rWalkMin, index = timeline).cumsum()
            rWalkMax = (rWalkMax * _pv * initSize).round(2)
            rWalkMin = (rWalkMin * _pv * initSize).round(2)
            self.foretrack_f0["min"] = concat((self.foretrack_f0["min"], rWalkMin),
                      axis = "columns").min(axis = "columns", numeric_only = True)
            self.foretrack_f0["max"] = concat((self.foretrack_f0["max"], rWalkMax),
                      axis = "columns").max(axis = "columns", numeric_only = True)

        self.foretrack_f0 += initFunds
        self.foretrack_f0["mid"] = self.foretrack_f0[["min", "max"]].mean(axis = "columns")
        profits = self.foretrack_f0.diff().fillna(0)
        growths = (profits / self.foretrack_f0.shift(1))
        growths = (growths.shift(1).fillna(0) + 1).cumprod()
        self.foretrack_f1 = initFunds + (profits * growths).cumsum()

        with PlotStyle.context(
            kwargs.get("style", self._defStyle)):
                self.foreplot = self.__forePlot(**kwargs)

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def __modTitle(self, t1: Timestamp, t2: Timestamp):
        
        title = self.title
        cut = title.split("@")[1]
        cut = cut.split(":")[1][: -1]
        cut = cut.split(" ")[: -1]
        isDate = lambda x: (len(x) > 2)
        tReplace = dict(zip(filter(isDate, cut),
            [t1.strftime("%Y/%m/%d"), t2.strftime("%Y/%m/%d")]))
        for sb, sa in tReplace.items(): title = title.replace(sb, sa)
        return title

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def __forePlot(self, **kwargs) -> dict():

        print("\nDrawing foretrack...")
        progBar = ProgBar(width = 40, verbose = "Drawing", items = [
            "figure", "title", "axes", "projections", "x-grid", "y-grid"])

        Fig = Figure()
        save = kwargs.pop("save", False)
        width, height = kwargs.get("figsize", (15, 10))
        Fig.set_figheight(height); Fig.set_figwidth(width)
        progBar.show()

        fSize = kwargs.pop("fontsize", 12)
        start, until = self.foretrack_f1.index[[0, -1]]
        title = self.__modTitle(start, until)
        title = "Foretrack - " + title
        underlined = title + "\n" + "‾‾‾" * len(title)
        Fig.suptitle(underlined, fontsize = int(fSize * 1.1))
        Fig.supxlabel(self.footnote, fontsize = fSize)
        progBar.show()

        Ax2 = Fig.add_subplot(5, 1, (5, 5))
        Ax1 = Fig.add_subplot(5, 1, (1, 4), sharex = Ax2)
        Fig.set_tight_layout((0, 0.15, 1, 0.70))
        timeline = datetime(self.foretrack_f0.index)
        progBar.show()

        Ax1.plot(timeline, self.foretrack_f0["mid"], lw = 3, 
            color = "skyblue", alpha = 0.8, label = "Avg, simple")
        Ax1.plot(timeline, self.foretrack_f1["mid"], lw = 3,
            color = "limegreen", alpha = 0.8, label = "Avg, compound")
        Ax1.fill_between(x = timeline, color = "blue", alpha = 0.4, lw = 0.0,
            y1 = self.foretrack_f0["min"], y2 = self.foretrack_f0["max"])
        Ax1.fill_between(x = timeline, color = "green", alpha = 0.4, lw = 0.0,
            y1 = self.foretrack_f1["min"], y2 = self.foretrack_f1["max"])
        
        Ax1.set_xticklabels([], color = "none")
        progBar.show()

        retMin = self.foretrack_f1["min"].pct_change().fillna(0)
        cretMin = numpy.exp(numpy.log(1 + retMin).cumsum())
        drawdown = 1 - cretMin / cretMin.cummax()
        Ax2.fill_between(x = drawdown.index, y1 = - drawdown * 100,
            y2 = drawdown * 0, color = "red", label = "Worst drawdown")
        Ax2.set_ylim(-drawdown.max() * 100, 0)

        xTicks = numpy.arange(width * 50 / 15)
        xTicks = xTicks * (until - start) / len(xTicks)
        xTicks = datetime(start + xTicks)
        Ax2.set_xticks(xTicks)
        xTicks = xTicks.strftime("%Y-%m-%d %H:%M")
        Ax2.set_xticklabels(xTicks, rotation = 90)
        Ax2.set_xlim(start, until)
        progBar.show()

        yMin0 = self.foretrack_f0["min"].min()
        yMin1 = self.foretrack_f1["min"].min()
        yMax0 = self.foretrack_f0["max"].max()
        yMax1 = self.foretrack_f1["max"].max()
        yMin, yMax = min(yMin0, yMin1), max(yMax0, yMax1)
        yTicks = TickRange.lin(yMin, yMax, maxLength = 20)
        Ax1.set_yticks(yTicks)
        Ax1.set_ylim(yTicks[0], yTicks[-1])
        progBar.show()

        Ax1.minorticks_off()
        Ax2.minorticks_off()
        Fig.set_tight_layout((0, 0, 1, 1))
        print("\nForetrack finished!")
        if save:
            print("\nSaving foretrack...")
            self.__save(Fig, title)
            print("...Foretrack saved!")
        return {"balance": Ax1, "drawdown": Ax2}

#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

if (__name__ == "__main__"):

    #display(Statement.fromBacktestMQL4(htmlFile = "backtest MT4.htm", specFile = "Specs IC Markets.csv", toJSON = False))
    #display(Statement.fromBacktestMQL4(htmlFile = "backtest MT4.htm", specFile = "Specs IC Markets.csv", toJSON = True))
    #display(Statement.fromHistoryMQL4(htmlFile = "statement MT4.htm", specFile = "Specs IC Markets.csv", toJSON = False))
    #display(Statement.fromHistoryMQL4(htmlFile = "statement MT4.htm", specFile = "Specs IC Markets.csv", toJSON = True))

    filename = "LiveProfitKU LegoMarkets 2020.07.29-2022.04.05.json"
    record = Record.fromJson(filename)
    record.distribute(save = True)
    # record.backtrack(sizeByEquity = False, save = True)
    # record.foretrack(until = "2023-01-01", iterations = 1000, save = True, outPropFreq = 10)