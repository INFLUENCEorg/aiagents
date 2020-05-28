# aiagents
Repository for Influence project agents


### Using the library ###

To use the library, simply add the checkout directory to your PYTHONPATH.
You can verify that it succeeded by typing 
```
import aiagents
```
in the python3 console.
Among other dependencies, you will need to have aienvs installed, or on your PYTHONPATH
https://github.com/INFLUENCEorg/aienvs
No build is needed.


### Build the library ###
If you would like build the library, 
* build aienvs first
* go to the aiagents directory and execute 
```
./build.sh
```
(or sudo ./build.sh depending on your user privileges)


# Using from Eclipse #
If you have the aienvs project installed, you can also add a dependency to that project directly. You also need to configure your eclipse interpreter as described on https://github.com/INFLUENCEorg/aienvs

If you want to use aiagents independently from a venv using pydev, prepare the venv as described. 
Additionally, pip install tensorflow.
