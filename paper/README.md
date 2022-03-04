The following command can be run from the repository root to compile a PDF of 
`paper.md` with the JOSS template:
```
docker run --rm --volume $PWD/paper:/data --user $(id -u):$(id -g) \
--env JOURNAL=joss openjournals/paperdraft
```