# Contributing

If you want to contribute to `LenslessPiCam and make it better, your help is
very welcome. Contributing is also a great way to learn more about the package
itself.

## Ways to contribute
- File bug reports, or suggest improvements / feature requests by opening an "Issue".
- Opening a "Pull request" (see below).


## Coding style

We use [Black](https://github.com/psf/black) to format the code. 

First install the library:
```bash
pip install black
```

You can then run the formatting script we've prepared:
```bash
./format_code.sh
```

## Unit tests

As much as possible, it is good practice to write unit tests to make sure 
code does not break when adding new functionality. We have prepared a few 
which can be run with [pytest](https://docs.pytest.org).

First install the library:
```
pip install pytest
```
And then run
```
pytest test/
```
to run all tests.

## How to make a clean pull request (PR)

1. Create a personal fork of the project.
2. Clone the fork on your local machine. 
3. Create a new branch to work on. Give it a new that reflects the bug / feature you will work on. It is recommended to keep the main branch "clean" and in sync with the original repository's main branch!
4. Implement / fix your feature, comment your code.
5. Write or adapt tests as needed.
6. Add or change the documentation as needed.
7. Format the code (see above).
8. Push your new branch to your fork.
9. Open a pull request with the original repository.
