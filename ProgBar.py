import sys

class ProgBar:

    def __init__(self, items: list, width: int = 40, verbose: str = ""):
        """
        A simple single-line progress bar to display during for-loops or long routines which may
        require visual tracking and patience. Note: may duplicate below if "`print`" function is
        called within the loop.\n
        Inputs:\n
            -> "`items`": a list of string tags over which the loop may be iterating. Usually it
            consists on what's right next to the "`in`" keyword. Can also be a number when the
            iterative argument is numeric (see examples).\n
            -> "`width`": an integer holding the amount of squares wide that the progress bar will
            have. Should be smaller than console/terminal/shell.\n
            -> "`verbose`": a string with a process descriptor. See examples.\n
        Examples:\n
            1)  >> `items = ["Step 1", "Step 2", "Step 3", "Step 4", "Step 5"]`\n
                >> `progBar = ProgBar(items = items, width = 40, verbose = "Doing")`\n
                >> `for item in items:`\n
                >> >> {do stuff}\n
                >> >> `progBar.show()`\n
            ...should print:\n
                >> `[========                                ] 25% Doing: Step 1`\n
            2)  >> `progBar = ProgBar(items = 5, width = 40, verbose = "Finished")`\n
                >> `for item in range(5):`\n
                >> >> {do stuff}\n
                >> >> `progBar.show()`\n
            ...should print:\n
                >> `[========                                ] 25% Finished: 1/5`\n
        """
        self.__verbose, self.__width, self.__prog = verbose, width, 0
        if not isinstance(items, int):
            self.__length = len(items)
            self.__items = list(items) + ["OK\n"]
            chars = max(map(len, self.__items))
            add = lambda x: x.ljust(1 + chars - len(x))
            self.__items = list(map(add, self.__items))
        else: self.__length, self.__items = items, list()
        self.show()

    def reset(self): self.__prog = 0 ; print("\n")

    def show(self, step: int = 1):
        """
        Function to update the progress bar. Should be added as a first or last line of the loop,
        depending on the case. See examples of "`ProgBar`" init method for further information.\n
        Inputs:\n
            -> "`step`" should be equal to 1 in most cases because progress bar shall be updated once
            every loop. But in case that multiple steps are to be considered at once, or if the loop
            shall conditionally skip one step, the number can be as larger as necessary.\n
        Outputs: None
        """
        if not self.__items: item = self.__prog
        else: item = self.__items[self.__prog]
        if self.__verbose: verbose = "%s: %s" % (self.__verbose, item)
        if not self.__items: verbose += " / " + str(self.__length)
        track = int(self.__width * self.__prog / self.__length)
        bar = "=" * int(track) + " " * int(self.__width - track)
        pc = 100 * track // self.__width
        sys.stdout.write("\r[%s] %d%% | %s" % (bar, pc, verbose))
        if (self.__prog > self.__length): self.reset()
        sys.stdout.flush() ; self.__prog += step