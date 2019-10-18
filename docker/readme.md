TODO: this document needs to be modified for users of the `aiagents` and `aienvs` repositories.


To install on cluster run the following, this will install sumo and sumoai in a dockerfile

```
git clone https://yourgituser:yourgitpw@github.com/INFLUENCEorg/sumoai.git

cd sumoai/docker

docker build -t=sumoaidocker -f=Dockerfile --build-arg GITUSER=yourgituser --build-arg GITPW=yourgitpw --build-arg GPU="TRUE" .
```
If not running on a GPU machine don't pass GPU="TRUE"

Then we can remove sumoai

```
cd ..
cd ..
rm -R sumoai
```

If using a GPU run a container with:
```
docker run -it --runtime=nvidia sumoaidocker:latest
```

If not:
```
docker run -it sumoaidocker:latest
```
