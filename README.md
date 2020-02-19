# aiagents
Repository for Influence project agents


# Prepare aiagents for use
To use this , you need to have aienvs installed. To install that, 
* Start up your aiagents venv
* import aienvs from the repo
* run the aienvs/build.sh script.

After that, you can use aiagents


### Build the library ###
To build the library, 
* follow the steps in 'prepare aiagents for use'
* go to the aiagents directory and execute 
```
./build.sh
```
(or sudo ./build.sh depending on your user privileges)


# Using from Eclipse #
If you have the aienvs project installed, you can also add a dependency to that project directly. You also need to configure your eclipse interpreter as described on https://github.com/INFLUENCEorg/aienvs

If you want to use aiagents independently from a venv using pydev, prepare the venv as described. 
Additionally, pip install tensorflow.
