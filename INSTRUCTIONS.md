# NLP Homework 4: Parsing

We assume that you've made a local copy of
<http://www.cs.jhu.edu/~jason/465/hw-parse/> (for example, by downloading
and unpacking the zipfile there) and that you are currently in that directory.

## QUESTION 3.

First try out the recognizer that we provided.

	./recognize.py papa.gr papa.sen
	./recognize.py -v papa.gr papa.sen    # verbose output

Copy the program `recognize.py` to `parse.py`.
Then edit `parse.py` to transform our unweighted recognizer 
into your probabilistic parser.  You should be able to call
it as

	./parse.py papa.gr papa.sen | ./prettyprint

Notice that the docstring for the `Agenda` class includes some sample calls and
their intended output.  The bottom of the script uses
[`doctest`](https://pymotw.com/2/doctest/) to check that those calls behave as
intended.  You may want to use `doctest` (or
[`unittest`](https://pymotw.com/2/unittest/)) to document and test your own
classes and methods, too.

You are free to ignore part or all of `recognize.py` and just write your own solution from scratch, if you prefer.  If your solution is not in Python, you should still submit a `parse.py`, which can just simply run your program, for example like this:

    #!/usr/bin/env python3
    import sys
    import subprocess
    subprocess.run(["java","-jar","myparser.jar"]  # invoke my parser
	                + sys.argv[1:])                # on the same args

----------

## QUESTION 4.

Notice that even `recognize` is very slow on a large grammar,
and of course `parse` will be at least as slow:

	./recognize.py --progress wallstreet.gr wallstreet.sen
	./parse.py --progress wallstreet.gr wallstreet.sen

So copy `parse.py` to `parse2.py` and make that version faster.

(The `--progress` option displays the fraction of columns
that have been processed so far.  In addition, the `-v` option reports 
the number of PREDICT, SCAN, and ATTACH actions for each sentence.
You could enhance it to report more detailed statistics if you like.)
