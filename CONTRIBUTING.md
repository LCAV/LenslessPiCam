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

## Release new version and deploy to PyPi

After merging to the `main` branch and from the `main` branch (!):

1. Edit the `version` field in `setup.py`.
2. Create new tag.
    ```
    git tag -a vX.X.X -m "Description."
    git push origin vX.X.X
3. Create package and upload to Pypi (`pip install twine` if not already done).
    ```
    python setup.py sdist
    python -m twine upload  dist/lensless-X.X.X.tar.gz
    ```
4. On [GitHub](https://github.com/LCAV/LenslessPiCam/tags) set 
the new tag by (1) clicking "..." and selecting "Create release" and (2) at the bottom pressing "Publish release".
5. Update `CHANGELOG.md` with new release version.
